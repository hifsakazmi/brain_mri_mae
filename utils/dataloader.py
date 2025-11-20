import os
import zipfile
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.mri_dataset import MRIDataset
import config

# For Kaggle download
import kagglehub

def prepare_dataset1():
    
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
        path = config.DATASET1_TRAIN_PATH if split=="train" else config.DATASET1_TEST_PATH
        num_classes = config.DATASET1_NUM_CLASSES

    elif dataset_name.lower() == "dataset2":
        local_path = prepare_dataset2()
        # Depending on how the Kaggle dataset is structured, adjust train/test split
        # Here we assume all images in subfolders; you may want to add a split
        path = os.path.join(local_path, "Training") if split=="train" else os.path.join(local_path, "Testing")
        num_classes = config.DATASET2_NUM_CLASSES

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dataset = MRIDataset(path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)
    return loader, num_classes
