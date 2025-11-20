from utils.dataloader import get_dataloader
import config
import matplotlib.pyplot as plt
import torch

def show_batch(images, labels, classes):
    """Display a batch of images with their labels"""
    batch_size = len(images)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
    if batch_size == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Convert from PyTorch tensor (C,H,W) to matplotlib format (H,W,C)
        img = images[i].permute(1, 2, 0).cpu()  # Change from (3,224,224) to (224,224,3)
        img = img.squeeze()  # remove extra dimensions if any
        
        ax.imshow(img, cmap="gray")
        ax.set_title(classes[labels[i]])
        ax.axis("off")
    plt.show()

def test_dataset(dataset_name="dataset1", split="train"):
    print(f"Testing {dataset_name} - {split} split")
    loader, num_classes = get_dataloader(dataset_name=dataset_name, split=split)
    
    # Get class names from MRIDataset
    classes = loader.dataset.class_names
    print(f"Found classes: {classes}")
    print(f"Number of samples: {len(loader.dataset)}")

    # Show a single batch
    for images, labels in loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels: {labels}")
        show_batch(images[:4], labels[:4], classes)  # show first 4 images
        break

if __name__ == "__main__":
    # Test Dataset 1 (binary)
    test_dataset("dataset1", "train")

    # Test Dataset 2 (4-class)
    test_dataset("dataset2", "train")
