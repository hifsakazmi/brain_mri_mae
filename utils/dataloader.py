import os
import zipfile
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets.mri_dataset import MRIDataset
import config

import kagglehub
import gdown

def prepare_dataset1():
    """
    Downloads and extracts the 2-class brain tumor dataset from Google Drive
    if it hasn't been downloaded yet.
    Returns local folder path containing the dataset.
    """
    local_path = "/content"  # This is where your files get extracted
    zip_path = "/content/images_binary.zip"
    
    # Check if the training_images folder exists
    training_path = "/content/training_images"
    testing_path = "/content/testing_images"
    
    if not os.path.exists(training_path) or not os.path.exists(testing_path):
        print("Downloading 2-class dataset from Google Drive...")
        
        # Install gdown if not already installed
        try:
            import gdown
        except ImportError:
            print("Installing gdown...")
            os.system("pip install -U --no-cache-dir gdown --pre")
            import gdown
        
        # Download the file
        print("Downloading zip file...")
        gdown.download(config.DATASET1_PATH, zip_path, quiet=False)
        
        # Check if download was successful
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Download failed! Zip file not found at {zip_path}")
        
        # Extract the zip file
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
        
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Dataset 1 already exists!")
    
    return local_path

def prepare_dataset2():
    """
    Downloads and extracts the 4-class Kaggle brain tumor dataset
    if it hasn't been downloaded yet.
    Returns local folder path containing the dataset.
    """
    local_path = f"/kaggle/input/{config.DATASET2_NAME}"
    if not os.path.exists(local_path):
        print("Downloading 4-class Kaggle dataset...")
        localpath = kagglehub.dataset_download(config.DATASET2_KAGGLE_PATH)
    return local_path


def get_dataloader(dataset_name="dataset1", split="train", batch_size=None, num_workers=None, img_size=None):
    """
    Returns a DataLoader for the requested dataset and split.
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    img_size = img_size or config.IMG_SIZE

    # Enhanced transforms with augmentation for training
    if split == "train":
        transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Only safe augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    if dataset_name.lower() == "dataset1":
        local_path = prepare_dataset1()
        path = os.path.join(local_path, "training_images") if split=="train" else os.path.join(local_path, "testing_images")
        num_classes = config.DATASET1_NUM_CLASSES

    elif dataset_name.lower() == "dataset2":
        local_path = prepare_dataset2()
        path = os.path.join(local_path, "Training") if split=="train" else os.path.join(local_path, "Testing")
        num_classes = config.DATASET2_NUM_CLASSES

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dataset = MRIDataset(path, transform)
    # Handle different splits

    if split == "test":
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"✅ Test dataset: {len(dataset)} samples")
    else: 
        # Split training data into 80% train, 20% validation
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        if split == "train":
            train_dataset.class_names = dataset.class_names  
            loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            print(f"Train dataset: {len(train_dataset)} samples (80% of training data)")
        else:  # val
            val_dataset.class_names = dataset.class_names
            loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            print(f"Validation dataset: {len(val_dataset)} samples (20% of training data)")

    return loader, num_classes
