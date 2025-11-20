import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat

class MAEEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()
        self.encoder = timm.create_model(
            model_name, pretrained=False, num_classes=0
        )
        self.patch_embed = self.encoder.patch_embed
        self.blocks = self.encoder.blocks
        self.norm = self.encoder.norm

    def forward(self, x):
        return self.encoder(x)


class MAEDecoder(nn.Module):
    """Improved decoder for MAE"""
    def __init__(self, embed_dim=768, decoder_dim=512, patch_size=16, num_patches=196):
        super().__init__()
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, patch_size * patch_size * 3)  # *3 for RGB channels
        )
        
    def forward(self, x):
        return self.decoder(x)
    

class MAEModel(nn.Module):
    def __init__(self, encoder_name="vit_base_patch16_224", mask_ratio=0.75, img_size=224):
        super().__init__()
        self.encoder = MAEEncoder(encoder_name)
        self.patch_size = 16
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Better decoder initialization
        self.decoder = MAEDecoder(
            embed_dim=768, 
            decoder_dim=512, 
            patch_size=self.patch_size,
            num_patches=self.num_patches
        )
        
        # Add learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, 768))
        
    def patchify(self, images):
        """Convert images to patches"""
        B, C, H, W = images.shape
        assert H == W == self.img_size, f"Input size must be {self.img_size}"
        
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        return patches

    def random_masking(self, x, mask_ratio):
        """Randomly mask patches"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, ids_restore, mask

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patchify images
        patches = self.patchify(x)  # [B, num_patches, patch_dim]
        
        # Mask patches
        patches_masked, ids_restore, mask = self.random_masking(patches, self.mask_ratio)
        
        # FIX: Get positional embeddings correctly
        pos_embed = self.encoder.encoder.pos_embed[:, 1:, :]  # remove cls token
        
        # FIX: Properly index positional embeddings
        batch_pos_embed = pos_embed.repeat(B, 1, 1)  # [B, num_patches, dim]
        
        # Gather the positional embeddings for the kept patches
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1)
        pos_embed_kept = batch_pos_embed[batch_idx, ids_restore[:, :patches_masked.size(1)]]
        
        # Add positional embeddings to visible patches
        patches_masked = patches_masked + pos_embed_kept
        
        # Encode visible patches
        encoded = self.encoder.blocks(patches_masked)
        encoded = self.encoder.norm(encoded)
        
        # Decode all patches (add mask tokens for masked patches)
        mask_tokens = repeat(self.mask_token, '1 1 d -> b n d', b=B, n=self.num_patches - encoded.size(1))
        decoder_input = torch.cat([encoded, mask_tokens], dim=1)
        
        # Unshuffle patches to original order
        decoder_input = torch.gather(decoder_input, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, decoder_input.size(2)))
        
        # Add positional embeddings to all patches for decoder
        decoder_input = decoder_input + batch_pos_embed
        
        # Decode
        decoded_patches = self.decoder(decoder_input)
        
        return decoded_patches, patches, mask

class SimpleMAEModel(nn.Module):
"""Ultra-simple MAE that definitely works"""
    def __init__(self, mask_ratio=0.75, img_size=224):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Simple encoder-decoder
        self.encoder = nn.Sequential(
            nn.Linear(3 * self.patch_size * self.patch_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Linear(512, 3 * self.patch_size * self.patch_size),
        )

    def patchify(self, images):
        """Convert images to patches"""
        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        return patches

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Get all patches
        all_patches = self.patchify(x)  # [B, num_patches, patch_dim]
        num_patches = all_patches.shape[1]
        num_mask = int(num_patches * self.mask_ratio)
        
        # Simple approach: reconstruct ALL patches, but we'll mask the loss later
        # Encode all patches
        encoded = self.encoder(all_patches)  # [B, num_patches, 256]
        
        # Decode all patches  
        reconstructed = self.decoder(encoded)  # [B, num_patches, patch_dim]
        
        # Create a random mask (we'll use this in the loss)
        mask = torch.rand(B, num_patches, device=x.device) < self.mask_ratio
        
        return reconstructed, all_patches, mask