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
        dataset_name="dataset1",  
        split="test",
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
            loss = (loss * mask).sum() / mask.sum()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def pretrain(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ProperMAE(
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MAE_LEARNING_RATE) 

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 

    # Create directory for saving models
    os.makedirs(os.path.dirname(cfg.MAE_ENCODER_SAVE_PATH), exist_ok=True)
    
    # For MAE pretraining, we don't need labels, so we can use any dataset
    train_loader, _ = get_dataloader(
        dataset_name="dataset1",  
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
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
            
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
        
        print(f"Epoch {epoch+1} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.encoder.state_dict(), cfg.MAE_ENCODER_SAVE_PATH)
            torch.save(model.state_dict(), cfg.MAE_FULL_SAVE_PATH)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to {cfg.MAE_ENCODER_SAVE_PATH}")

if __name__ == "__main__":
    import config
    pretrain(config)