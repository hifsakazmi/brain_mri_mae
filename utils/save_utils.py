# utils/save_utils.py
import os
import torch
from google.colab import drive
from datetime import datetime

def mount_drive():
    """Mount Google Drive if not already mounted"""
    try:
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive mounted")
    except Exception as e:
        print(f"⚠️  Drive mount failed (might already be mounted): {e}")

def save_to_drive(model, model_name, drive_path="/content/drive/MyDrive/brain_mri_mae/models"):
    """Save model to Google Drive"""
    # Mount Drive
    mount_drive()
    
    # Create directory
    os.makedirs(drive_path, exist_ok=True)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(drive_path, f"{model_name}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to Google Drive: {save_path}")
    
    return save_path

def save_encoder_to_drive(encoder, model_name, drive_path="/content/drive/MyDrive/brain_mri_mae/models"):
    """Save only encoder to Google Drive"""
    mount_drive()
    os.makedirs(drive_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(drive_path, f"{model_name}_encoder_{timestamp}.pth")
    torch.save(encoder.state_dict(), save_path)
    print(f"✅ Encoder saved to Google Drive: {save_path}")
    
    return save_path