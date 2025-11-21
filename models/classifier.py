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
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through encoder layers
        x = self.encoder_layers(x)
        
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
            
        return self.classifier(x)