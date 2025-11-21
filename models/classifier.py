import torch
import torch.nn as nn
import torch.nn.functional as F

# In classifier.py - FIX THIS
class MAEClassifier(nn.Module):
    def __init__(self, encoder, num_classes, img_size=224, patch_size=16, encoder_dim=768, use_cls_token=True, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.use_cls_token = use_cls_token
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        
        # Use encoder's components directly
        self.patch_embed = encoder.patch_embed
        self.pos_embed = encoder.pos_embed
        
        # Classification head
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
        # Patch embedding using encoder's method
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # Add CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through encoder 
        x = self.encoder(x)  
        
        # Classification
        if self.use_cls_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Global average pooling
            
        return self.classifier(x)