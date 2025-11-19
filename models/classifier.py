import torch
import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    def __init__(self, pretrained_encoder_path=None, num_classes=3):
        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )

        if pretrained_encoder_path:
            state = torch.load(pretrained_encoder_path, map_location="cpu")
            self.model.load_state_dict(state, strict=False)

    def forward(self, x):
        return self.model(x)
