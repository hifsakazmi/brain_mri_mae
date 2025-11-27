import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_mae import MAEModel
from utils.dataloader import get_dataloader 

def validate_mae(model, cfg, device):
    """Validate MAE model on test split"""
    model.eval()
    val_loader, _ = get_dataloader(
        dataset_name=cfg.DATASET,  
        split="val",
        batch_size=cfg.MAE_BATCH_SIZE,
        num_workers=2 
    )
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            decoded_patches, original_patches, mask = model(imgs)
            
            # Same loss calculation as training
            loss = (decoded_patches - original_patches) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-5) 
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def debug_loss_calculation(model, cfg, device):
    """Debug the loss calculation"""
    model.eval()
    loader, _ = get_dataloader(cfg.DATASET, "train", cfg.MAE_BATCH_SIZE, num_workers=2)
    
    with torch.no_grad():
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)
        decoded_patches, original_patches, mask = model(imgs)
        
        print(f"Input images shape: {imgs.shape}")
        print(f"Decoded patches shape: {decoded_patches.shape}")
        print(f"Original patches shape: {original_patches.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask sum (how many masked): {mask.sum().item()}")
        print(f"Mask ratio: {mask.sum().item() / mask.numel():.3f}")
        
        # Check individual components
        diff = (decoded_patches - original_patches) ** 2
        print(f"Diff range: [{diff.min().item():.6f}, {diff.max().item():.6f}]")
        
        loss_per_patch = diff.mean(dim=-1)
        print(f"Loss per patch range: [{loss_per_patch.min().item():.6f}, {loss_per_patch.max().item():.6f}]")
        
        masked_loss = (loss_per_patch * mask).sum()
        mask_sum = mask.sum()
        print(f"Masked loss sum: {masked_loss.item():.6f}")
        print(f"Mask sum: {mask_sum.item()}")
        
        final_loss = masked_loss / mask_sum
        print(f"Final loss: {final_loss.item():.6f}")
        
        # Check if patches are actually different
        mse_between = torch.nn.functional.mse_loss(decoded_patches, original_patches)
        print(f"Direct MSE between decoded and original: {mse_between.item():.6f}")

def pretrain(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = MAEModel(
        img_size=cfg.MAE_IMG_SIZE,
        patch_size=cfg.MAE_PATCH_SIZE,
        encoder_dim=cfg.MAE_ENCODER_DIM,
        encoder_depth=cfg.MAE_ENCODER_DEPTH,
        encoder_heads=cfg.MAE_ENCODER_HEADS,
        decoder_dim=cfg.MAE_DECODER_DIM,
        decoder_depth=cfg.MAE_DECODER_DEPTH,
        decoder_heads=cfg.MAE_DECODER_HEADS,
        mask_ratio=cfg.MAE_MASK_RATIO
    ).to(device)

    # DEBUG: Check loss calculation
    print("=== DEBUGGING LOSS CALCULATION ===")
    debug_loss_calculation(model, cfg, device)
    print("=== END DEBUG ===")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MAE_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 

    # Create local directories
    os.makedirs(os.path.dirname(cfg.MAE_ENCODER_SAVE_PATH), exist_ok=True)
    
    # Import save utilities
    from utils.save_utils import save_best_models, is_drive_mounted
    
    # Check Drive status
    if is_drive_mounted():
        print("✅ Google Drive is mounted - models will be saved to Drive")
    else:
        print("⚠️  Google Drive not mounted - models will be saved locally")
    
    # Initialize loss tracking
    train_losses = []
    val_losses = []
    epochs_list = []

    # For MAE pretraining, we don't need labels, so we can use any dataset
    train_loader, _ = get_dataloader(
        dataset_name=cfg.DATASET,  
        split="train",
        batch_size=cfg.MAE_BATCH_SIZE,
        num_workers=2 
    )

    # Track best model
    best_val_loss = float('inf')
    
    model.train()
    for epoch in range(cfg.MAE_EPOCHS): 
        total_loss = 0
        num_batches = 0
        
        # Training
        for batch_idx, (imgs, _) in enumerate(train_loader):  
            imgs = imgs.to(device)
            decoded_patches, original_patches, mask = model(imgs)
            
            loss = (decoded_patches - original_patches) ** 2
            loss = loss.mean(dim=-1)  # [B, N] - mean over patch dimensions
            loss = (loss * mask).sum() / (mask.sum() + 1e-5)  # Only masked patches
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_loss = total_loss / num_batches
        
        # Validation
        val_loss = validate_mae(model, cfg, device)
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_list.append(epoch + 1)
        
        print(f"Epoch {epoch+1} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step()  # Update learning rate
        
        # Save best model (to Drive if mounted, otherwise locally)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save using our smart save function
            success = save_best_models(
                encoder=model.encoder,
                full_model=model,
                model_name="mae_proper",
                local_encoder_path=cfg.MAE_ENCODER_SAVE_PATH,
                local_full_path=cfg.MAE_FULL_SAVE_PATH
            )
            
            if success:
                print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
            else:
                print(f"⚠️  Model saved with warnings. Val Loss: {val_loss:.4f}")

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    # Plot loss curves
    plot_loss_curves(epochs_list, train_losses, val_losses)
    
    # Save loss data
    save_loss_data(epochs_list, train_losses, val_losses)

def plot_loss_curves(epochs, train_losses, val_losses):
    """Plot training and validation loss curves"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MAE Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    
    # Save plot
    plt.savefig('./training_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_loss_data(epochs, train_losses, val_losses):
    """Save loss data to file for later analysis"""
    import json
    import pandas as pd
    
    loss_data = {
        'epochs': epochs,
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses]
    }
    
    # Save as JSON
    with open('./training_loss_data.json', 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    df.to_csv('./training_loss_data.csv', index=False)
    
    print("✅ Loss data saved to './training_loss_data.json' and './training_loss_data.csv'")


if __name__ == "__main__":
    import config
    pretrain(config)