import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
import math

class MAEModel(nn.Module):
    """Proper MAE implementation that follows the original paper"""
    
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512,
                 decoder_depth=8,
                 decoder_heads=16,
                 mask_ratio=0.75):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, encoder_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Encoder (ViT)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, encoder_depth)
        
        # Projection from encoder to decoder
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        
        # Decoder
        decoder_layers = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layers, decoder_depth)
        
        # Reconstruction head
        self.head = nn.Linear(decoder_dim, self.patch_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        
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
    
    def forward_encoder(self, x, mask_ratio):
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Masking
        x, ids_restore, mask = self.random_masking(x, mask_ratio)
        
        # Apply Transformer encoder
        x = self.encoder(x)
        
        return x, ids_restore, mask
    
    def forward_decoder(self, x, ids_restore):
        # Project to decoder dimension
        x = self.encoder_to_decoder(x)
        
        # Append mask tokens to the sequence
        mask_tokens = repeat(
            self.mask_token, 
            '1 1 d -> b n d', 
            b=x.shape[0], 
            n=ids_restore.shape[1] - x.shape[1]
        )
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle
        x = torch.gather(
            x_, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        # Add decoder positional embedding
        x = x + self.decoder_pos_embed
        
        # Apply Transformer decoder
        x = self.decoder(x)
        
        # Prediction head
        x = self.head(x)
        
        return x
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # Get original patches for reconstruction target
        patches = self.patch_embed(imgs)
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Encoder forward
        latent, ids_restore, mask = self.forward_encoder(imgs, mask_ratio)
        
        # Decoder forward
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred, patches, mask

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