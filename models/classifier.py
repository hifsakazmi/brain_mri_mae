import torch
import torch.nn as nn
import torch.nn.functional as F

class MAEClassifier(nn.Module):
    def __init__(self, encoder, num_classes, img_size=224, patch_size=16, encoder_dim=768, use_cls_token=True, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.use_cls_token = use_cls_token
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        
        # Create patch embedding if not present in encoder
        if not hasattr(encoder, 'patch_embed'):
            self.patch_embed = nn.Conv2d(
                3, encoder_dim, 
                kernel_size=patch_size, 
                stride=patch_size
            )
            # Create positional embeddings
            num_patches = (img_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.patch_embed = encoder.patch_embed
            self.pos_embed = encoder.pos_embed
        
        # Classification head
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
            self.classifier = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
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
        if hasattr(self.encoder, 'patch_embed'):
            x = self.encoder.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.encoder.pos_embed
        else:
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_embed
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through encoder blocks
        x = self.encoder(x)  # Direct forward pass
        
        # Classification
        if self.use_cls_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)
            
        return self.classifier(x)