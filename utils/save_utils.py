# utils/save_utils.py
import os
import torch
from datetime import datetime

def is_drive_mounted():
    """Check if Google Drive is already mounted"""
    return os.path.exists('/content/drive/MyDrive')

def save_model_with_fallback(model, model_name, local_path=None, drive_path="/content/drive/MyDrive/brain_mri_mae/models"):
    """Save model to Drive if mounted, otherwise save locally"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_drive_mounted():
        try:
            # Save to Google Drive
            os.makedirs(drive_path, exist_ok=True)
            drive_save_path = os.path.join(drive_path, f"{model_name}_{timestamp}.pth")
            torch.save(model.state_dict(), drive_save_path)
            print(f"✅ Model saved to Google Drive: {drive_save_path}")
            return drive_save_path
        except Exception as e:
            print(f"❌ Failed to save to Google Drive: {e}")
            # Fall back to local save
            return save_model_locally(model, model_name, local_path, timestamp)
    else:
        # Save locally
        return save_model_locally(model, model_name, local_path, timestamp)

def save_model_locally(model, model_name, local_path, timestamp):
    """Save model locally"""
    try:
        if local_path:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            torch.save(model.state_dict(), local_path)
            print(f"✅ Model saved locally: {local_path}")
            return local_path
        else:
            # Create a local backup path
            local_backup = f"./models/{model_name}_{timestamp}.pth"
            os.makedirs(os.path.dirname(local_backup), exist_ok=True)
            torch.save(model.state_dict(), local_backup)
            print(f"✅ Model saved locally: {local_backup}")
            return local_backup
    except Exception as e:
        print(f"❌ Failed to save locally: {e}")
        return None

def save_best_models(encoder, full_model, model_name, local_encoder_path, local_full_path):
    """Save both encoder and full model with Drive fallback"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    encoder_saved = False
    full_saved = False
    
    # Save encoder
    if encoder is not None:
        encoder_path = save_model_with_fallback(
            encoder, 
            f"{model_name}_encoder",
            local_encoder_path
        )
        encoder_saved = encoder_path is not None
    else:
        print("❌ Encoder is None, skipping save")
    
    # Save full model
    if full_model is not None:
        full_path = save_model_with_fallback(
            full_model,
            f"{model_name}_full", 
            local_full_path
        )
        full_saved = full_path is not None
    else:
        print("❌ Full model is None, skipping save")
    
    return encoder_saved and full_saved

def save_classifier_to_drive(classifier, drive_path="/content/drive/MyDrive/brain_mri_mae/models/classifier.pth"):
    """Save classifier to Google Drive"""
    try:
        os.makedirs(os.path.dirname(drive_path), exist_ok=True)
        torch.save(classifier.state_dict(), drive_path)
        print(f"✅ Classifier saved to Google Drive: {drive_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save classifier to Drive: {e}")
        return False