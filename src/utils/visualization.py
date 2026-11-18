import os
import numpy as np
import matplotlib.pyplot as plt

def save_batch_visualization(dataloader, save_path="patch_viz.png", batch_index=0, slice_idx=None):
    """
    Extracts a single 3D patch from the dataloader, takes a 2D slice, 
    and saves the EM and Label visualization to disk.
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The DataLoader to sample from.
    save_path : str
        File path to save the generated PNG image.
    batch_index : int
        Which item in the batch to visualize (default 0).
    slice_idx : int or None
        The Z-slice to visualize. If None, calculates the middle slice dynamically.
    """
    # 1. Fetch a single batch
    em_batch, lbl_batch = next(iter(dataloader))
    
    print(f"Sampled EM Batch Shape:    {em_batch.shape} | Type: {em_batch.dtype}")
    print(f"Sampled Label Batch Shape: {lbl_batch.shape} | Type: {lbl_batch.dtype}")
    
    # 2. Extract specific item
    # Shape of em_batch is usually [B, C, Z, Y, X], lbl is [B, Z, Y, X]
    em_tensor = em_batch[batch_index]
    lbl_tensor = lbl_batch[batch_index]
    
    # Convert to numpy and drop the channel dimension for EM
    em_single = em_tensor.squeeze(0).cpu().numpy()
    lbl_single = lbl_tensor.cpu().numpy()
    
    # 3. Determine the slice to plot
    z_dim = em_single.shape[0]
    mid_z = slice_idx if slice_idx is not None else z_dim // 2
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
    
    axes[0].imshow(em_single[mid_z], cmap='gray')
    axes[0].set_title(f"EM Patch (Slice {mid_z})", color='white')
    axes[0].axis('off')
    
    # Mask out -1 unannotated regions to 0 for visualization purposes
    visual_lbl = np.where(lbl_single[mid_z] == -1, 0, lbl_single[mid_z])
    
    # Use vmin and vmax to lock the colormap to the 30 classes so colors remain consistent
    im = axes[1].imshow(visual_lbl, cmap='nipy_spectral', interpolation='nearest', vmin=0, vmax=29)
    axes[1].set_title(f"Label Patch (Slice {mid_z})", color='white')
    axes[1].axis('off')
    
    # Crosshairs to mark center
    for ax in axes:
        ax.axhline(mid_z, color='blue', linestyle='-', linewidth=1)
        ax.axvline(mid_z, color='blue', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    # 5. Save and properly close the figure to free RAM
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    
    print(f"Saved batch visualization to: {save_path}")