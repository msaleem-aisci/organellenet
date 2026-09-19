"""
Unified modular training loop for BlueMind OrganelleNet.
Features: AMP, 1CycleLR, Stateful Checkpointing, and robust Early Stopping.
"""

import os
import sys
import gc
import csv
import shutil
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast            # Updated AMP import
from torch.cuda.amp import GradScaler

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

        self.num_epochs = config.training.num_epochs
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        self.mixed_precision = config.training.mixed_precision
        
        self.patience = config.training.early_stopping_patience
        self.patience_counter = 0
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = GradScaler(enabled=self.mixed_precision)

        self.checkpoint_dir = config.path.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.latest_ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
        self.best_ckpt_path = os.path.join(self.checkpoint_dir, "model_best.pth")
        self.log_file = os.path.join(self.checkpoint_dir, "training_log.csv")

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.print_freq = config.training.print_freq

    def _train_epoch(self, dataloader, epoch, scheduler):
        self.model.train()
        epoch_loss = 0.0
        steps_per_epoch = len(dataloader)
        
        self.optimizer.zero_grad(set_to_none=True)

        for step, (em_batch, lbl_batch) in enumerate(dataloader):
            em_batch = em_batch.to(self.device)
            
            if lbl_batch.dim() == 4 and lbl_batch.shape[1] == 1:
                lbl_batch = lbl_batch.squeeze(1)
            lbl_batch = lbl_batch.to(self.device, dtype=torch.long)
            
            # Updated AMP Syntax for modern PyTorch
            with autocast(device_type='cuda', enabled=self.mixed_precision):
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
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for step, (em_batch, lbl_batch) in enumerate(dataloader):
                em_batch = em_batch.to(self.device)
                
                if lbl_batch.dim() == 4 and lbl_batch.shape[1] == 1:
                    lbl_batch = lbl_batch.squeeze(1)
                lbl_batch = lbl_batch.to(self.device, dtype=torch.long)
                
                # Updated AMP Syntax
                with autocast(device_type='cuda', enabled=self.mixed_precision):
                    outputs = self.model(em_batch)
                    val_loss = self.criterion(outputs, lbl_batch, current_epoch=epoch)
                    
                epoch_loss += val_loss.item()
                
        return epoch_loss / len(dataloader)

    def _save_checkpoint(self, epoch, avg_val_loss, scheduler):
        is_best = avg_val_loss < self.best_val_loss
        
        if is_best:
            print(f">>> Validation loss improved from {self.best_val_loss:.4f} to {avg_val_loss:.4f}. Saving BEST model.")
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0 
            torch.save(self.model.state_dict(), self.best_ckpt_path)
        else:
            self.patience_counter += 1
            print(f">>> No improvement in val_loss. Early Stopping Patience: {self.patience_counter}/{self.patience}")
        
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

    def train(self, train_dataloader, val_dataloader, args, resume=True):
        """The main execution pipeline orchestrating the training lifecycle."""
        
        scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=self.lr, 
            epochs=self.num_epochs, 
            steps_per_epoch=len(train_dataloader),
            pct_start=0.05
        )

        # 1. Determine Checkpoint Load Path
        load_path = None
        if args.resume_ckpt and os.path.exists(args.resume_ckpt):
            load_path = args.resume_ckpt
            print(f"[*] Manual checkpoint path provided: {load_path}")
        elif resume and os.path.exists(self.latest_ckpt_path):
            load_path = self.latest_ckpt_path

        # 2. Handle CSV Logs Copier
        log_mode = "w"
        if args.resume_logs and os.path.exists(args.resume_logs):
            shutil.copyfile(args.resume_logs, self.log_file)
            log_mode = "a"  # Switch to append mode since the file now exists locally

        # 3. Load Checkpoint State
        if load_path:
            print(f"[*] Resuming model state from {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Conditionally load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                old_state = checkpoint['scheduler_state_dict']
                if old_state.get('total_steps') == scheduler.total_steps:
                    scheduler.load_state_dict(old_state)
                else:
                    print(f"[!] Epoch config extended. Generating a fresh OneCycleLR curve for the remaining epochs.")
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.patience_counter = checkpoint.get('patience_counter', 0)
            
            # If we are doing a standard local resume and didn't copy a file, set to append
            if not args.resume_logs:
                log_mode = "a"
                
            print(f"[*] Successfully restored state. Resuming at Epoch {self.start_epoch}. Current Patience: {self.patience_counter}")
        else:
            print("[*] Starting training from scratch (Epoch 0).")

        # 4. Initialize or Append Log File
        with open(self.log_file, log_mode, newline='') as f:
            writer = csv.writer(f)
            if log_mode == "w":
                writer.writerow(["epoch", "lr", "train_loss", "val_loss"])

        # 5. Main Training Loop
        for epoch in range(self.start_epoch, self.num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n=== Epoch [{epoch+1}/{self.num_epochs}] | Starting LR: {current_lr:.2e} ===")
            
            train_loss = self._train_epoch(train_dataloader, epoch, scheduler)
            val_loss = self._validate_epoch(val_dataloader, epoch)
            
            print(f"Epoch [{epoch}] Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            with open(self.log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, current_lr, train_loss, val_loss])
                
            self._save_checkpoint(epoch, val_loss, scheduler)
            
            gc.collect()
            torch.cuda.empty_cache()

            if self.patience_counter >= self.patience:
                print(f"\n[!] Early Stopping triggered at Epoch {epoch}. Validation loss has not improved for {self.patience} epochs.")
                break

        print("\nTraining complete.")