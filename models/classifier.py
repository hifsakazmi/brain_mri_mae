# models/classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAEClassifier(nn.Module):
    def __init__(self, encoder, num_classes, use_cls_token=True, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.use_cls_token = use_cls_token
        self.hidden_dim = hidden_dim
        
        # Get encoder output dimension from patch_embed
        encoder_dim = encoder.patch_embed.weight.shape[1]  # [embed_dim, in_chans, patch_size, patch_size]
        
        # Classification head
        if use_cls_token:
            # Add CLS token for classification
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
            # Global average pooling
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
        x = self.encoder.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add positional embedding
        x = x + self.encoder.pos_embed
        
        if self.use_cls_token:
            # Add CLS token
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through encoder blocks
        for block in self.encoder.encoder.layers:
            x = block(x)
        
        # Classification
        if self.use_cls_token:
            # Use CLS token for classification
            x = x[:, 0]  # CLS token
        else:
            # Global average pooling
            x = x.mean(dim=1)
            
        return self.classifier(x)

class SimpleMAEClassifier(nn.Module):
    """Classifier for SimpleMAE encoder"""
    def __init__(self, encoder, num_classes, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),  # SimpleMAE encoder outputs 256-dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Get patches
        patches = self.encoder.get_pixel_patches(x)
        # Encode
        features = self.encoder.encoder(patches)
        # Global average pooling
        features = features.mean(dim=1)
        return self.classifier(features)