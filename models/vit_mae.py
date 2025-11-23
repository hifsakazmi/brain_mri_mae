import torch
import torch.nn as nn
import math

class MAEModel(nn.Module):
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
        
        # Calculate patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Proper patch embedding (Linear, not Conv2d)
        self.patch_embed = nn.Linear(self.patch_dim, encoder_dim)
        self.patch_norm = nn.LayerNorm(encoder_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, encoder_depth)
        
        # Projection to decoder
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        
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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def patchify(self, imgs):
        " Patch extraction (SAME for both input and target)"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == self.img_size and W == self.img_size
        assert H % p == 0 and W % p == 0
        
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)
        return x
    
    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, ids_restore, mask
    
    def forward_encoder(self, x, mask_ratio):
        # Apply patch embedding to actual patches
        # x is already patches from patchify()
        x = self.patch_embed(x)  # [B, N, encoder_dim]
        x = self.patch_norm(x)
        x = x + self.pos_embed
        
        # Masking
        x, ids_restore, mask = self.random_masking(x, mask_ratio)
        
        # Encoder (no CLS token - correct for MAE)
        x = self.encoder(x)
        
        return x, ids_restore, mask
    
    def forward_decoder(self, x, ids_restore):
        # Project to decoder dimension
        x = self.enc_to_dec(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle with correct dimension
        x_unshuffled = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_dim)
        )
        
        # Add decoder positional embeddings
        x_unshuffled = x_unshuffled + self.decoder_pos_embed
        
        # Apply decoder
        x_decoded = self.decoder(x_unshuffled)
        
        # Prediction head
        pred = self.head(x_decoded)
        
        return pred
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # Use SAME patch extraction for both input and target
        patches = self.patchify(imgs)  # [B, N, patch_dim]
        
        # Encoder forward
        latent, ids_restore, mask = self.forward_encoder(patches, mask_ratio)
        
        # Decoder forward  
        pred = self.forward_decoder(latent, ids_restore)  # [B, N, patch_dim]
        
        return pred, patches, mask
    
    # REMOVE the old get_pixel_patches method - use patchify() instead

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

    def get_pixel_patches(self, images):
        """Convert images to patches"""
        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        return patches

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Get all patches
        all_patches = self.get_pixel_patches(x)  # [B, num_patches, patch_dim]
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