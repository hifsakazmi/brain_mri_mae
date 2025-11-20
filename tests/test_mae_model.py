# tests/test_mae_model.py

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_mae import MAEModel, SimpleMAEModel
from utils.dataloader import get_dataloader
import config

def test_proper_mae_forward():
    """Test ProperMAE forward pass and shapes"""
    print("=== TESTING PROPER MAE FORWARD PASS ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MAEModel(
        img_size=config.MAE_IMG_SIZE,
        patch_size=config.MAE_PATCH_SIZE,
        encoder_dim=config.MAE_ENCODER_DIM,
        encoder_depth=config.MAE_ENCODER_DEPTH,
        encoder_heads=config.MAE_ENCODER_HEADS,
        decoder_dim=config.MAE_DECODER_DIM,
        decoder_depth=config.MAE_DECODER_DEPTH,
        decoder_heads=config.MAE_DECODER_HEADS,
        mask_ratio=config.MAE_MASK_RATIO
    ).to(device)
    
    # Test with dummy data
    dummy_imgs = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        pred, patches, mask = model(dummy_imgs)
        pixel_patches = model.get_pixel_patches(dummy_imgs)
        
        print(f"Input shape: {dummy_imgs.shape}")
        print(f"Predicted patches: {pred.shape}")
        print(f"Embedding patches: {patches.shape}")
        print(f"Pixel patches: {pixel_patches.shape}")
        print(f"Mask: {mask.shape}")
        
        # Critical check: pred and pixel_patches should have same shape for loss
        assert pred.shape == pixel_patches.shape, f"Shape mismatch: {pred.shape} vs {pixel_patches.shape}"
        print("✅ Shape check passed!")
        
        # Test loss calculation
        loss = (pred - pixel_patches) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        print(f"Test loss: {loss.item():.6f}")
        
    print("=== PROPER MAE TEST COMPLETED ===\n")

def test_simple_mae_forward():
    """Test SimpleMAE forward pass"""
    print("=== TESTING SIMPLE MAE FORWARD PASS ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SimpleMAEModel(
        mask_ratio=config.MAE_MASK_RATIO,
        img_size=config.MAE_IMG_SIZE
    ).to(device)
    
    dummy_imgs = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        reconstructed, original, mask = model(dummy_imgs)
        
        print(f"Input shape: {dummy_imgs.shape}")
        print(f"Reconstructed: {reconstructed.shape}")
        print(f"Original: {original.shape}")
        print(f"Mask: {mask.shape}")
        
        # Shapes should match
        assert reconstructed.shape == original.shape, "Shape mismatch!"
        print("✅ Shape check passed!")
        
        # Test loss
        loss = (reconstructed - original) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        print(f"Test loss: {loss.item():.6f}")
        
    print("=== SIMPLE MAE TEST COMPLETED ===\n")

def test_mae_with_real_data():
    """Test MAE models with real dataset"""
    print("=== TESTING WITH REAL DATASET ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test both models
    models = {
        "ProperMAE": MAEModel(...).to(device),  # your config
        "SimpleMAE": SimpleMAEModel(...).to(device)
    }
    
    # Get real data
    loader, _ = get_dataloader(
        dataset_name=config.DATASET,
        split="train", 
        batch_size=2,
        num_workers=2
    )
    
    for name, model in models.items():
        print(f"Testing {name}...")
        model.eval()
        with torch.no_grad():
            imgs, _ = next(iter(loader))
            imgs = imgs.to(device)
            
            if name == "ProperMAE":
                pred, _, mask = model(imgs)
                target = model.get_pixel_patches(imgs)
            else:
                pred, target, mask = model(imgs)
            
            # Check loss
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
            print(f"{name} loss: {loss.item():.6f}")
    
    print("=== REAL DATA TEST COMPLETED ===")

if __name__ == "__main__":
    test_proper_mae_forward()
    test_simple_mae_forward() 
    test_mae_with_real_data()