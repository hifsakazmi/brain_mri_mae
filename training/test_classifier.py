
# test_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import MAEClassifier
from utils.dataloader import get_dataloader
import config

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

def plot_test_roc_curves(metrics, num_classes, class_names):
    """Plot ROC curves for test set"""
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'][:, 1])
        roc_auc = roc_auc_score(metrics['labels'], metrics['probabilities'][:, 1])
        plt.plot(fpr, tpr, color='darkred', lw=2, 
                label=f'Test ROC (AUC = {roc_auc:.3f})')
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
    plt.title('Test Set - ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./test_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_test_confusion_matrix(metrics, class_names):
    """Plot confusion matrix for test set"""
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Set - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('./test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_best_classifier_model():
    """Find the best classifier model from possible locations"""
    possible_paths = [
        config.CLASSIFIER_SAVE_PATH,
        "/content/drive/MyDrive/brain_mri_mae/models/classifier.pth",
        "./models/classifier.pth",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found classifier at: {path}")
            return path
    
    print("❌ No classifier model found in any expected location!")
    return None

def test_classifier(cfg, dataset_name, classifier_path=None):
    """Load best classifier and perform comprehensive testing on test set"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Testing on dataset: {dataset_name}")
    
    # Get test dataset
    test_loader, num_classes = get_dataloader(
        dataset_name=dataset_name,
        split="test",
        batch_size=cfg.CLASSIFIER_BATCH_SIZE,
        num_workers=2
    )
    
    # Get class names
    class_names = test_loader.dataset.class_names
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model architecture (same as during training)
    from models.vit_mae import MAEModel
    
    # Create MAE model structure (for encoder)
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
    
    # Create classifier with same architecture
    classifier = MAEClassifier(
        mae_model.encoder,
        num_classes,
        img_size=cfg.MAE_IMG_SIZE,
        patch_size=cfg.MAE_PATCH_SIZE,
        encoder_dim=cfg.MAE_ENCODER_DIM
    )
    
    # Manually set patch embedding (same as training)
    classifier.patch_embed = nn.Conv2d(
        3, cfg.MAE_ENCODER_DIM,
        kernel_size=cfg.MAE_PATCH_SIZE, 
        stride=cfg.MAE_PATCH_SIZE
    )
    
    classifier = classifier.to(device)
    
    # Find the best classifier model
    if classifier_path is None:
        classifier_path = find_best_classifier_model()
    
    if classifier_path is None:
        print("❌ Please specify a classifier path or train a classifier first.")
        return None
    
    print(f"🔄 Loading classifier from: {classifier_path}")
    
    try:
        # Load the trained classifier
        checkpoint = torch.load(classifier_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['state_dict'])
        else:
            classifier.load_state_dict(checkpoint)
            
        print("✅ Classifier loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return None
    
    # Perform comprehensive testing
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    test_metrics = compute_metrics(classifier, test_loader, device, num_classes)
    
    # Calculate test loss
    classifier.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    # Print comprehensive results
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"F1 Score:      {test_metrics['f1_score']:.4f}")
    print(f"Precision:     {test_metrics['precision']:.4f}")
    print(f"Recall:        {test_metrics['recall']:.4f}")
    print(f"AUC-ROC:       {test_metrics['auc_roc']:.4f}")
    
    # Detailed classification report
    print("\n" + "="*40)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(test_metrics['labels'], test_metrics['predictions'], 
                              target_names=class_names, digits=4))
    
    # Per-class metrics
    print("\n" + "="*40)
    print("PER-CLASS METRICS")
    print("="*40)
    if num_classes > 2:
        precision_per_class = precision_score(test_metrics['labels'], test_metrics['predictions'], average=None)
        recall_per_class = recall_score(test_metrics['labels'], test_metrics['predictions'], average=None)
        f1_per_class = f1_score(test_metrics['labels'], test_metrics['predictions'], average=None)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:15} | Precision: {precision_per_class[i]:.4f} | Recall: {recall_per_class[i]:.4f} | F1: {f1_per_class[i]:.4f}")
    
    # Generate test-specific plots
    print("\n📊 Generating test set visualizations...")
    plot_test_roc_curves(test_metrics, num_classes, class_names)
    plot_test_confusion_matrix(test_metrics, class_names)
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'accuracy': float(test_metrics['accuracy']),
        'f1_score': float(test_metrics['f1_score']),
        'precision': float(test_metrics['precision']),
        'recall': float(test_metrics['recall']),
        'auc_roc': float(test_metrics['auc_roc']),
        'class_names': class_names,
        'num_classes': num_classes,
        'test_samples': len(test_loader.dataset),
        'predictions': test_metrics['predictions'].tolist(),
        'labels': test_metrics['labels'].tolist(),
        'probabilities': test_metrics['probabilities'].tolist()
    }
    
    import json
    with open('./test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n✅ Test evaluation completed!")
    print("📁 Results saved to:")
    print("   - test_results.json")
    print("   - test_roc_curves.png") 
    print("   - test_confusion_matrix.png")
    
    return test_metrics

if __name__ == "__main__":
    print("Starting Classifier Testing...")
    test_metrics = test_classifier(config, dataset_name=config.DATASET)
    
    if test_metrics:
        print(f"\n FINAL TEST RESULTS:")
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   F1 Score:  {test_metrics['f1_score']:.4f}")
        print(f"   AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    else:
        print("❌ Testing failed!")