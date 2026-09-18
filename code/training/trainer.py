"""
Unified modular training loop for BlueMind OrganelleNet.
Features: AMP, 1CycleLR, Stateful Checkpointing, and robust Early Stopping.
"""

import os
import sys
import gc
import csv
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class Trainer:
    def __init__(self, model, criterion, config, device, patience=10):
        self.model = model.to(device)
        self.criterion = criterion
        self.config = config
        self.device = device

        # 1. Hyperparameters 
        self.num_epochs = config.training.num_epochs
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        self.mixed_precision = config.training.mixed_precision
        
        # Early Stopping Configuration
        # Safely pull from config if it exists, otherwise use the default passed in __init__
        self.patience = getattr(config.training, "early_stopping_patience", patience)
        self.patience_counter = 0
        
        # 2. Optimizer and Scaler
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = GradScaler(enabled=self.mixed_precision)

        # 3. Paths
        self.checkpoint_dir = config.path.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.latest_ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
        self.best_ckpt_path = os.path.join(self.checkpoint_dir, "model_best.pth")
        self.log_file = os.path.join(self.checkpoint_dir, "training_log.csv")

        # 4. State variables
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.print_freq = 50 

    def _train_epoch(self, dataloader, epoch, scheduler):
        """Handles the forward pass, backward pass, and optimizer steps for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        steps_per_epoch = len(dataloader)
        
        self.optimizer.zero_grad(set_to_none=True)

        for step, (em_batch, lbl_batch) in enumerate(dataloader):
            em_batch = em_batch.to(self.device)
            
            if lbl_batch.dim() == 4 and lbl_batch.shape[1] == 1:
                lbl_batch = lbl_batch.squeeze(1)
            lbl_batch = lbl_batch.to(self.device, dtype=torch.long)
            
            with autocast(enabled=self.mixed_precision):
                outputs = self.model(em_batch)
                loss = self.criterion(outputs, lbl_batch, current_epoch=epoch)
                
            self.scaler.scale(loss).backward()
            
            epoch_loss += loss.item()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            scheduler.step()
                
            if (step + 1) % self.print_freq == 0:
                print(f"  [Train] Step {step+1}/{steps_per_epoch} | Loss: {loss.item():.4f}")
                
        return epoch_loss / steps_per_epoch

    def _validate_epoch(self, dataloader, epoch):
        """Handles model evaluation for one epoch without gradient tracking."""
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for step, (em_batch, lbl_batch) in enumerate(dataloader):
                em_batch = em_batch.to(self.device)
                
                if lbl_batch.dim() == 4 and lbl_batch.shape[1] == 1:
                    lbl_batch = lbl_batch.squeeze(1)
                lbl_batch = lbl_batch.to(self.device, dtype=torch.long)
                
                with autocast(enabled=self.mixed_precision):
                    outputs = self.model(em_batch)
                    val_loss = self.criterion(outputs, lbl_batch, current_epoch=epoch)
                    
                epoch_loss += val_loss.item()
                
        return epoch_loss / len(dataloader)

    def _save_checkpoint(self, epoch, avg_val_loss, scheduler):
        """Manages saving state, tracking best model, and updating early stopping."""
        is_best = avg_val_loss < self.best_val_loss
        
        if is_best:
            print(f">>> Validation loss improved from {self.best_val_loss:.4f} to {avg_val_loss:.4f}. Saving BEST model.")
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0  # Reset early stopping counter
            torch.save(self.model.state_dict(), self.best_ckpt_path)
        else:
            self.patience_counter += 1
            print(f">>> No improvement in val_loss. Early Stopping Patience: {self.patience_counter}/{self.patience}")
        
        # CRITICAL: Save the patience_counter so early stopping does not reset on resume
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter 
        }
        torch.save(checkpoint, self.latest_ckpt_path)

    def train(self, train_dataloader, val_dataloader, resume=True):
        """The main execution pipeline orchestrating the training lifecycle."""
        
        scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=self.lr, 
            epochs=self.num_epochs, 
            steps_per_epoch=len(train_dataloader),
            pct_start=0.05
        )

        # 1. Robust Resume Logic
        log_mode = "w"
        if resume and os.path.exists(self.latest_ckpt_path):
            print(f"[*] Resuming from {self.latest_ckpt_path}")
            checkpoint = torch.load(self.latest_ckpt_path, map_location=self.device, weights_only=True)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            
            # Safely restore the early stopping state (default to 0 if from an older checkpoint)
            self.patience_counter = checkpoint.get('patience_counter', 0)
            
            log_mode = "a"
            print(f"[*] Successfully restored state. Resuming at Epoch {self.start_epoch}. Current Patience: {self.patience_counter}")
        else:
            print("[*] Starting training from scratch (Epoch 0).")

        # 2. Setup Logging
        with open(self.log_file, log_mode, newline='') as f:
            writer = csv.writer(f)
            if log_mode == "w":
                writer.writerow(["epoch", "lr", "train_loss", "val_loss"])

        # 3. Main Loop
        for epoch in range(self.start_epoch, self.num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n=== Epoch [{epoch}/{self.num_epochs-1}] | Starting LR: {current_lr:.2e} ===")
            
            # Execute phases cleanly
            train_loss = self._train_epoch(train_dataloader, epoch, scheduler)
            val_loss = self._validate_epoch(val_dataloader, epoch)
            
            print(f"Epoch [{epoch}] Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            with open(self.log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, current_lr, train_loss, val_loss])
                
            self._save_checkpoint(epoch, val_loss, scheduler)
            
            # Memory Cleanup
            gc.collect()
            torch.cuda.empty_cache()

            # 4. Early Stopping Trigger
            if self.patience_counter >= self.patience:
                print(f"\n[!] Early Stopping triggered at Epoch {epoch}. Validation loss has not improved for {self.patience} epochs.")
                break

        print("\nTraining complete.")