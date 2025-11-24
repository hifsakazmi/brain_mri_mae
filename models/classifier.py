import torch
import torch.nn as nn
import torch.nn.functional as F

# classifier.py 
# IN classifier.py - REPLACE THE ENTIRE CLASS:

class MAEClassifier(nn.Module):
    def __init__(self, encoder, num_classes, img_size=224, patch_size=16, encoder_dim=768, use_cls_token=True, hidden_dim=512, dropout=0.1):
        super().__init__()
        
        # Check if we got full MAE model or just encoder
        if hasattr(encoder, 'patch_embed'):
            # Full MAE model passed
            self.patch_embed = encoder.patch_embed
            self.pos_embed = encoder.pos_embed
            self.encoder_layers = encoder.encoder  # The actual TransformerEncoder
        else:
            # Just encoder passed (TransformerEncoder)
            self.patch_embed = nn.Conv2d(
                3, encoder_dim, 
                kernel_size=patch_size, 
                stride=patch_size
            )
            num_patches = (img_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.encoder_layers = encoder
        
        self.use_cls_token = use_cls_token
        self.encoder_dim = encoder_dim
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add positional embedding
        B, N, D = x.shape
        
        # Handle positional embedding size mismatch
        if self.pos_embed.shape[1] != N:
            # This can happen if image size doesn't match expected size
            # Interpolate the positional embedding
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2), 
                size=N, 
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        # Add CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through encoder layers
        if isinstance(self.encoder_layers, nn.TransformerEncoder):
            x = self.encoder_layers(x)
        else:
            # If it's a custom encoder, make sure it expects the right input
            x = self.encoder_layers(x)
        
        # Classification token or global average pooling
        if self.use_cls_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Global average pooling
            
        return self.classifier(x)