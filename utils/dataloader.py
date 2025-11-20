import os
import zipfile
from torch.utils.data import DataLoader
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
    
    if not os.path.exists("/content/training_images") or not os.path.exists("/content/testing_images"):
        print("Downloading 2-class dataset from Google Drive...")
        
        # Install gdown if not already installed
        try:
            import gdown
        except ImportError:
            print("Installing gdown...")
            os.system("pip install -U --no-cache-dir gdown --pre")
            import gdown
        
        # Download the file
        gdown.download(config.DATASET1_PATH, zip_path, quiet=False)
        
        # Extract the zip file
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
        
        # Remove the zip file to save space
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully!")
    
    return local_path

def prepare_dataset2():
    """
    Downloads and extracts the 4-class Kaggle brain tumor dataset
    if it hasn't been downloaded yet.
    Returns local folder path containing the dataset.
    """
    local_path = "/kaggle/input/brain-tumor-classification-mri"
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

    # Define transform that converts images to tensors
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    if dataset_name.lower() == "dataset1":
        local_path = prepare_dataset1()
        path = os.path.join(local_path, "Training") if split=="train" else os.path.join(local_path, "Testing")
        num_classes = config.DATASET1_NUM_CLASSES

    elif dataset_name.lower() == "dataset2":
        local_path = prepare_dataset2()
        path = os.path.join(local_path, "Training") if split=="train" else os.path.join(local_path, "Testing")
        num_classes = config.DATASET2_NUM_CLASSES

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dataset = MRIDataset(path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)
    return loader, num_classes
