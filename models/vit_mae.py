import torch
import torch.nn as nn
import timm
from einops import rearrange

class MAEEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()
        self.encoder = timm.create_model(
            model_name, pretrained=False, num_classes=0
        )

    def forward(self, x):
        return self.encoder(x)


class MAEDecoder(nn.Module):
    """Simple decoder for MAE (can be improved)"""

    def __init__(self, embed_dim=768, decoder_dim=512, patch_size=16):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, patch_size * patch_size)
        )

    def forward(self, encoded_tokens):
        return self.decoder(encoded_tokens)
    

class MAEModel(nn.Module):
    def __init__(self, encoder_name="vit_base_patch16_224", mask_ratio=0.75):
        super().__init__()
        self.encoder = MAEEncoder(encoder_name)
        self.decoder = MAEDecoder()
        self.mask_ratio = mask_ratio
        self.patch_size = 16

    def random_masking(self, x):
        B, C, H, W = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        num_keep = int(num_patches * (1 - self.mask_ratio))

        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        return ids_keep, ids_shuffle

    def forward(self, x):
        ids_keep, ids_shuffle = self.random_masking(x)

        tokens = self.encoder.encoder.patch_embed(x)  # patch embeddings
        B, N, D = tokens.shape

        batch_idx = torch.arange(B).unsqueeze(-1)
        kept_tokens = tokens[batch_idx, ids_keep, :]

        encoded = self.encoder.encoder.blocks(kept_tokens)
        decoded = self.decoder(encoded)

        return decoded
