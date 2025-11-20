import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_mae import MAEModel
from models.vit_mae import SimpleMAEModel
from utils.dataloader import get_dataloader 

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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MAE_LEARNING_RATE) 

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 

    # Create directory for saving models
    os.makedirs(os.path.dirname(cfg.MAE_ENCODER_SAVE_PATH), exist_ok=True)
    
    # For MAE pretraining, we don't need labels, so we can use any dataset
    loader, _ = get_dataloader(
        dataset_name="dataset1",  
        split="train",
        batch_size=cfg.MAE_BATCH_SIZE,
        num_workers=2 
    )

    model.train()
    for epoch in range(cfg.MAE_EPOCHS): 
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (imgs, _) in enumerate(loader):  
            imgs = imgs.to(device)

            # Get all three returns from MAE model
            decoded_patches, original_patches, mask = model(imgs)
            
            # DEBUG: Print shapes to see the issue
            print(f"DEBUG - decoded_patches shape: {decoded_patches.shape}")
            print(f"DEBUG - original_patches shape: {original_patches.shape}") 
            print(f"DEBUG - mask shape: {mask.shape}")
            
            # Apply mask to focus loss only on reconstructed (masked) patches
            loss = (decoded_patches - original_patches) ** 2
            loss = loss.mean(dim=-1)  # Mean over patch dimensions
            loss = (loss * mask).sum() / mask.sum()  # Mean only over masked patches
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} — Average loss: {avg_loss:.4f}")

    torch.save(model.encoder.state_dict(), cfg.MAE_ENCODER_SAVE_PATH)
    torch.save(model.state_dict(), cfg.MAE_FULL_SAVE_PATH) 
    print(f"Model saved to {cfg.MAE_ENCODER_SAVE_PATH}")

if __name__ == "__main__":
    import config
    pretrain(config)