import torch
from torch.utils.data import DataLoader
from models.vit_mae import MAEModel
from datasets.mri_dataset import MRIDataset
from training.utils import load_unlabeled_paths

def pretrain(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MAEModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    unlabeled_paths = load_unlabeled_paths(config.UNLABELED_DIR)
    dataset = MRIDataset(unlabeled_paths)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(config.PRETRAIN_EPOCHS):
        for imgs in loader:
            imgs = imgs.to(device)
            decoded = model(imgs)

            loss = torch.nn.functional.mse_loss(decoded, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} — loss: {loss.item()}")

    torch.save(model.encoder.state_dict(), config.MAE_SAVE_PATH)
