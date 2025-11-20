# utils/save_utils.py
import os
import torch
from datetime import datetime

def is_drive_mounted():
    """Check if Google Drive is already mounted"""
    return os.path.exists('/content/drive/MyDrive')

def mount_drive_safe():
    """Mount Google Drive only if not already mounted"""
    if not is_drive_mounted():
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    else:
        print("✅ Google Drive already mounted")

def save_to_drive(model, model_name, drive_path="/content/drive/MyDrive/brain_mri_mae/models"):
    """Save model to Google Drive"""
    try:
        # Mount Drive safely
        mount_drive_safe()
        
        # Create directory
        os.makedirs(drive_path, exist_ok=True)
        
        # Save model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(drive_path, f"{model_name}_{timestamp}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to Google Drive: {save_path}")
        
        return save_path
    except Exception as e:
        print(f"❌ Failed to save to Google Drive: {e}")
        return None

def save_encoder_to_drive(encoder, model_name, drive_path="/content/drive/MyDrive/brain_mri_mae/models"):
    """Save only encoder to Google Drive"""
    try:
        mount_drive_safe()
        os.makedirs(drive_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(drive_path, f"{model_name}_encoder_{timestamp}.pth")
        torch.save(encoder.state_dict(), save_path)
        print(f"✅ Encoder saved to Google Drive: {save_path}")
        
        return save_path
    except Exception as e:
        print(f"❌ Failed to save encoder to Google Drive: {e}")
        return None

def save_best_to_drive(encoder, full_model, model_name, drive_path="/content/drive/MyDrive/brain_mri_mae/models"):
    """Save both encoder and full model to Drive"""
    try:
        mount_drive_safe()
        os.makedirs(drive_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save encoder
        encoder_path = os.path.join(drive_path, f"{model_name}_encoder_{timestamp}.pth")
        torch.save(encoder.state_dict(), encoder_path)
        
        # Save full model
        full_path = os.path.join(drive_path, f"{model_name}_full_{timestamp}.pth")
        torch.save(full_model.state_dict(), full_path)
        
        print(f"✅ Best models saved to Google Drive:")
        print(f"   Encoder: {encoder_path}")
        print(f"   Full: {full_path}")
        
        return encoder_path, full_path
    except Exception as e:
        print(f"❌ Failed to save best models to Google Drive: {e}")
        return None, None