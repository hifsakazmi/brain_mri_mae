import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MRIDataset(Dataset):
    def __init__(self, root_path, img_size=224):
        """
        root_path: local folder containing images organized by class subfolders
                   e.g., /content/training_images/no, /content/training_images/yes
                   or the downloaded 4-class Kaggle dataset
        img_size: resize all images to this size
        """
        self.root_path = root_path
        self.img_paths = []
        self.labels = []
        self.classes = []

        # Detect classes (subfolder names)
        for class_name in sorted(os.listdir(root_path)):
            class_path = os.path.join(root_path, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                for fname in os.listdir(class_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.img_paths.append(os.path.join(class_path, fname))
                        self.labels.append(class_name)

        # Map class names to integer labels
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]

        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale MRI images
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("L")  # convert to grayscale
        image = self.transform(image)
        return image, label
