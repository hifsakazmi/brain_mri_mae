# datasets/mri_dataset.py

import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    """
    A unified dataset loader for:
    1. Binary datasets where class is encoded in filename (yes@, no@)
    2. Multi-class datasets where class is derived from folder structure
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Decide dataset type:
        # Case A: subfolders = multi-class dataset (Kaggle 4-class)
        # Case B: flat files = label based on filename pattern

        subfolders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        if len(subfolders) > 0:
            # ------------------------------------------------
            # 4-class dataset (uses folder names as labels)
            # ------------------------------------------------
            self.class_names = sorted(subfolders)
            self.samples = []

            for idx, cls in enumerate(self.class_names):
                cls_folder = os.path.join(root, cls)
                img_files = glob.glob(os.path.join(cls_folder, "*.png")) + glob.glob(os.path.join(cls_folder, "*.jpg"))
                for path in img_files:
                    self.samples.append((path, idx))

        else:
            # ------------------------------------------------
            # 2-class dataset (filename contains "yes" or "no")
            # ------------------------------------------------
            self.class_names = ["no", "yes"]
            self.samples = []

            img_files = glob.glob(os.path.join(root, "*.png")) + glob.glob(os.path.join(root, "*.jpg"))

            for path in img_files:
                fname = os.path.basename(path).lower()
                if "yes" in fname:
                    label = 1
                else:
                    label = 0

                self.samples.append((path, label))


        print(f"[MRIDataset] Loaded {len(self.samples)} samples from {root}")
        print(f"[MRIDataset] Classes: {self.class_names}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
