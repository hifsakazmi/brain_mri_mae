# training/finetune_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import time
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import MAEClassifier
from utils.dataloader import get_dataloader
import config

def get_latest_drive_checkpoint(drive_path="/content/drive/MyDrive/brain_mri_mae/models"):
    """Get the latest MAE checkpoint from Google Drive without mounting"""
    try:
        if os.path.exists(drive_path):
            encoder_files = [f for f in os.listdir(drive_path) if "mae_proper_encoder" in f and f.endswith(".pth")]
            
            if not encoder_files:
                print("⚠️  No MAE encoder files found in Google Drive")
                return None
            
            encoder_files.sort(reverse=True)
            latest_encoder = os.path.join(drive_path, encoder_files[0])
            
            print(f"✅ Found latest MAE encoder in Drive: {encoder_files[0]}")
            return latest_encoder
        else:
            print("⚠️  Google Drive path not found, using local checkpoint")
            return None
            
    except Exception as e:
        print(f"⚠️  Error accessing Google Drive: {e}")
        return None

def load_pretrained_encoder(cfg, pretrained_path):
    """Load pre-trained MAE encoder from checkpoint"""
    from models.vit_mae import MAEModel
    
    try:
        print("🔄 Trying to load as ProperMAE...")
        mae_model = MAEModel(
            img_size=cfg.MAE_IMG_SIZE,
            patch_size=cfg.MAE_PATCH_SIZE,
            encoder_dim=cfg.MAE_ENCODER_DIM,
            encoder_depth=cfg.MAE_ENCODER_DEPTH,
            encoder_heads=cfg.MAE_ENCODER_HEADS,
            decoder_dim=cfg.MAE_DECODER_DIM,
            decoder_depth=cfg.MAE_DECODER_DEPTH, 
            decoder_heads=cfg.MAE_DECODER_HEADS,
            mask_ratio=cfg.MAE_MASK_RATIO
        )
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'patch_embed.weight' in checkpoint:
            mae_model.load_state_dict(checkpoint)
            encoder = mae_model.encoder
        else:
            encoder = mae_model.encoder
            encoder.load_state_dict(checkpoint)
            
        print("✅ Loaded ProperMAE encoder")
        return encoder, "proper"
        
    except Exception as e:
        print(f"❌ Failed to load as ProperMAE: {e}")


def compute_metrics(model, data_loader, device, num_classes):
    """Compute comprehensive metrics including AUC-ROC"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    
    # AUC-ROC (one-vs-rest for multi-class)
    if num_classes > 2:
        auc_roc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
    else:
        auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def plot_roc_curves(metrics, num_classes, class_names):
    """Plot ROC curves for all classes"""
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'][:, 1])
        roc_auc = roc_auc_score(metrics['labels'], metrics['probabilities'][:, 1])
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class classification
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(metrics['labels'], classes=range(num_classes))
        
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], metrics['probabilities'][:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], metrics['probabilities'][:, i])
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(metrics, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('./confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_history(train_metrics_history, val_metrics_history):
    """Plot training history for all metrics"""
    epochs = range(1, len(train_metrics_history) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    ax1.plot(epochs, [m['accuracy'] for m in train_metrics_history], 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, [m['accuracy'] for m in val_metrics_history], 'r-', label='Val Accuracy', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 Score
    ax2.plot(epochs, [m['f1_score'] for m in train_metrics_history], 'b-', label='Train F1', linewidth=2)
    ax2.plot(epochs, [m['f1_score'] for m in val_metrics_history], 'r-', label='Val F1', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision
    ax3.plot(epochs, [m['precision'] for m in train_metrics_history], 'b-', label='Train Precision', linewidth=2)
    ax3.plot(epochs, [m['precision'] for m in val_metrics_history], 'r-', label='Val Precision', linewidth=2)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # AUC-ROC
    ax4.plot(epochs, [m['auc_roc'] for m in train_metrics_history], 'b-', label='Train AUC-ROC', linewidth=2)
    ax4.plot(epochs, [m['auc_roc'] for m in val_metrics_history], 'r-', label='Val AUC-ROC', linewidth=2)
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('AUC-ROC')
    ax4.set_title('AUC-ROC')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./training_metrics_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    """Train for one epoch and return metrics"""
    model.train()
    running_loss = 0.0
    
    # Collect predictions for metrics
    all_predictions = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        # DEBUG: ADD GRADIENT HERE (first batch only)
        if batch_idx == 0:
            print("=== GRADIENT CHECK ===")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
                else:
                    print(f"{name}: NO GRADIENT")
            print("======================")
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predictions = outputs.max(1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    
    # Compute training metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    
    return epoch_loss, accuracy, f1, precision

def validate(model, val_loader, criterion, device, num_classes):
    """Validate the model and return comprehensive metrics"""
    return compute_metrics(model, val_loader, device, num_classes)

def finetune_classifier(cfg, dataset_name, use_drive_checkpoint=True):
    """Fine-tune classifier on pre-trained MAE encoder with comprehensive metrics"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Fine-tuning on dataset: {dataset_name}")
    
    # Get dataset info
    train_loader, num_classes = get_dataloader(
        dataset_name=dataset_name,
        split="train",
        batch_size=cfg.CLASSIFIER_BATCH_SIZE,
        num_workers=2
    )
    
    val_loader, _ = get_dataloader(
        dataset_name=dataset_name,
        split="val", 
        batch_size=cfg.CLASSIFIER_BATCH_SIZE,
        num_workers=2
    )
    
    # Get class names from the dataset
    class_names = train_loader.dataset.class_names
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create new MAE model and load weights into encoder
    from models.vit_mae import MAEModel
    
    # Create MAE model structure
    mae_model = MAEModel(
        img_size=cfg.MAE_IMG_SIZE,
        patch_size=cfg.MAE_PATCH_SIZE,
        encoder_dim=cfg.MAE_ENCODER_DIM,
        encoder_depth=cfg.MAE_ENCODER_DEPTH,
        encoder_heads=cfg.MAE_ENCODER_HEADS,
        decoder_dim=cfg.MAE_DECODER_DIM,
        decoder_depth=cfg.MAE_DECODER_DEPTH,
        decoder_heads=cfg.MAE_DECODER_HEADS,
        mask_ratio=cfg.MAE_MASK_RATIO
    )
    
    # Try to load pre-trained weights
    checkpoint_loaded = False
    checkpoint_path = None
    
    if use_drive_checkpoint:
        checkpoint_path = get_latest_drive_checkpoint()
    
    if checkpoint_path is None and os.path.exists(cfg.MAE_ENCODER_SAVE_PATH):
        checkpoint_path = cfg.MAE_ENCODER_SAVE_PATH
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"🔄 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'patch_embed.weight' in checkpoint:
                # Full MAE model checkpoint
                mae_model.load_state_dict(checkpoint)
            else:
                # Encoder-only checkpoint - load into encoder
                mae_model.encoder.load_state_dict(checkpoint)
            
            print("✅ Pre-trained weights loaded successfully")
            checkpoint_loaded = True
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("⚠️  Using randomly initialized weights")
    else:
        print("⚠️  No checkpoint found, using random initialization")
    
    # Use the encoder from the MAE model
    classifier = MAEClassifier(
        mae_model.encoder,  
        num_classes,
        img_size=cfg.MAE_IMG_SIZE,
        patch_size=cfg.MAE_PATCH_SIZE,
        encoder_dim=cfg.MAE_ENCODER_DIM
    )

    # Manually ensure patch embedding is correct
    classifier.patch_embed = nn.Conv2d(
        3, cfg.MAE_ENCODER_DIM,
        kernel_size=cfg.MAE_PATCH_SIZE, 
        stride=cfg.MAE_PATCH_SIZE
    )

    # Copy positional embedding if needed
    if hasattr(mae_model, 'pos_embed'):
        classifier.pos_embed = mae_model.pos_embed
    
    print(f"✅ Classifier created {'with pre-trained weights' if checkpoint_loaded else 'with random initialization'}")
    
    classifier = classifier.to(device)

    print("=== MODEL STRUCTURE DEBUG ===")
    print(f"Input shape: {cfg.MAE_IMG_SIZE}x{cfg.MAE_IMG_SIZE}")
    print(f"Patch embed weight shape: {classifier.patch_embed[0].weight.shape if isinstance(classifier.patch_embed, nn.Sequential) else classifier.patch_embed.weight.shape}")
    print("=============================")
    
    # Loss function
    #if dataset_name == "dataset2":
        # Apply class weights for 4-class imbalance
        # Weights computed with inverse frequency
        #class_weights = torch.tensor([1.63, 0.91, 0.87, 0.88]).to(device)
        #criterion = nn.CrossEntropyLoss()
    #else:
    criterion = nn.CrossEntropyLoss()
    
    # Enhanced fine-tuning: Differential learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in classifier.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'classifier' not in n],
            'weight_decay': cfg.CLASSIFIER_WEIGHT_DECAY,
            'lr': cfg.CLASSIFIER_LEARNING_RATE * 0.1  # Lower LR for backbone
        },
        {
            'params': [p for n, p in classifier.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'classifier' not in n],
            'weight_decay': 0.0,
            'lr': cfg.CLASSIFIER_LEARNING_RATE * 0.1
        },
        {
            'params': [p for n, p in classifier.named_parameters() if 'classifier' in n],
            'weight_decay': cfg.CLASSIFIER_WEIGHT_DECAY,
            'lr': cfg.CLASSIFIER_LEARNING_RATE  # Higher LR for classifier head
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    # Use cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training history
    train_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    best_val_acc = 0.0
    
    print("Starting fine-tuning...")
    for epoch in range(cfg.CLASSIFIER_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_acc, train_f1, train_precision = train_epoch(
            classifier, train_loader, criterion, optimizer, device, num_classes
        )
        
        # Validate with comprehensive metrics
        val_metrics = validate(classifier, val_loader, criterion, device, num_classes)
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        train_losses.append(train_loss)
        train_metrics_history.append({
            'accuracy': train_acc,
            'f1_score': train_f1,
            'precision': train_precision,
            'auc_roc': 0.0  # Not computed during training for efficiency
        })
        val_metrics_history.append(val_metrics)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{cfg.CLASSIFIER_EPOCHS} | Time: {epoch_time:.2f}s')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'  Val Acc: {val_metrics["accuracy"]:.4f} | Val F1: {val_metrics["f1_score"]:.4f} | Val AUC: {val_metrics["auc_roc"]:.4f}')
        print(f'  Val Precision: {val_metrics["precision"]:.4f} | Val Recall: {val_metrics["recall"]:.4f}')
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(classifier.state_dict(), cfg.CLASSIFIER_SAVE_PATH)

            # Save to Drive
            from utils.save_utils import save_classifier_to_drive
            save_classifier_to_drive(classifier)

            print(f'  ✅ New best model saved! Val Acc: {val_metrics["accuracy"]:.4f}')
        
        print('-' * 80)
    
    print(f'Training completed! Best validation accuracy: {best_val_acc:.4f}')
    
    # Final evaluation with best model
    print("\n" + "="*60)
    print("FINAL EVALUATION WITH BEST MODEL")
    print("="*60)
    
    # Load best model
    classifier.load_state_dict(torch.load(cfg.CLASSIFIER_SAVE_PATH))
    final_val_metrics = validate(classifier, val_loader, criterion, device, num_classes)
    
    # Print final metrics
    print(f"Accuracy:  {final_val_metrics['accuracy']:.4f}")
    print(f"F1 Score:  {final_val_metrics['f1_score']:.4f}")
    print(f"Precision: {final_val_metrics['precision']:.4f}")
    print(f"Recall:    {final_val_metrics['recall']:.4f}")
    print(f"AUC-ROC:   {final_val_metrics['auc_roc']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(final_val_metrics['labels'], final_val_metrics['predictions'], 
                              target_names=class_names))
    
    # Generate plots
    plot_metrics_history(train_metrics_history, val_metrics_history)
    plot_roc_curves(final_val_metrics, num_classes, class_names)
    plot_confusion_matrix(final_val_metrics, class_names)
    
    # Save final metrics to file
    final_metrics = {
        'accuracy': float(final_val_metrics['accuracy']),
        'f1_score': float(final_val_metrics['f1_score']),
        'precision': float(final_val_metrics['precision']),
        'recall': float(final_val_metrics['recall']),
        'auc_roc': float(final_val_metrics['auc_roc']),
        'class_names': class_names,
        'num_classes': num_classes
    }
    
    import json
    with open('./final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print("✅ All metrics and plots saved!")
    
    return final_val_metrics

if __name__ == "__main__":
    import config
    final_metrics = finetune_classifier(config, dataset_name=config.DATASET, use_drive_checkpoint=True)