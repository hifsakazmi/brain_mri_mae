import torch
from torch.utils.data import DataLoader
from models.classifier import ViTClassifier
from datasets.mri_dataset import MRIDataset
from training.utils import load_labeled_paths

def finetune(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViTClassifier(
        pretrained_encoder_path=config.MAE_SAVE_PATH,
        num_classes=config.NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    train_paths, train_labels = load_labeled_paths(config.TRAIN_DIR)
    val_paths, val_labels = load_labeled_paths(config.VAL_DIR)

    train_loader = DataLoader(
        MRIDataset(train_paths, train_labels),
        batch_size=16,
        shuffle=True
    )

    for epoch in range(config.FINETUNE_EPOCHS):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Fine-tune Epoch {epoch+1} — loss: {loss.item()}")
