# Modified for the ReCogDrive visual backbone ablation release.
"""Visual backbones for the ReCogDrive ablation experiments."""
from pathlib import Path
from types import SimpleNamespace

import timm
import torch
from safetensors.torch import load_file
from torch import nn
from torchvision import models


_TORCHVISION = {
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "fc", 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, "fc", 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, "fc", 2048),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, "heads", 768),
}
_TIMM = {
    "vit_large": ("vit_large_patch16_224", 1024, {"img_size": 224, "dynamic_img_size": True}),
    "vit_huge": ("vit_huge_patch14_224", 1280, {"img_size": 224, "dynamic_img_size": True}),
    "eva_clip_base": ("eva02_base_patch16_clip_224", 768, {}),
    "eva_clip_large": ("eva02_large_patch14_clip_224", 1024, {}),
}
BACKBONE_DIMS = {name: spec[-1] for name, spec in _TORCHVISION.items()}
BACKBONE_DIMS.update({name: spec[1] for name, spec in _TIMM.items()})
for alias, canonical in {"resnet": "resnet34", "vit": "vit_b_16", "vit_l_16": "vit_large"}.items():
    BACKBONE_DIMS[alias] = BACKBONE_DIMS[canonical]


class RecogDriveBackbone(nn.Module):
    def __init__(self, model_type, checkpoint_path="", device="cuda", trainable=False, pretrained=True):
        super().__init__()
        model_type = {"resnet": "resnet34", "vit": "vit_b_16", "vit_l_16": "vit_large"}.get(
            model_type, model_type
        )
        if model_type not in BACKBONE_DIMS:
            raise ValueError(f"Unknown backbone {model_type!r}. Choose from {', '.join(BACKBONE_DIMS)}.")
        self.trainable = trainable
        self.feature_dim = BACKBONE_DIMS[model_type]
        if model_type in _TORCHVISION:
            factory, weights, head, _ = _TORCHVISION[model_type]
            self.model = factory(weights=weights if pretrained and not checkpoint_path else None)
            setattr(self.model, head, nn.Identity())
        else:
            architecture, _, options = _TIMM[model_type]
            self.model = timm.create_model(
                architecture, pretrained=pretrained and not checkpoint_path, num_classes=0, **options
            )
        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.is_dir():
                files = (path / "model.safetensors", path / "pytorch_model.bin")
                path = next((file for file in files if file.is_file()), files[0])
            if path.suffix == ".safetensors":
                state = load_file(str(path))
            else:
                state = torch.load(path, map_location="cpu", weights_only=True)
            if "pos_embed" in state and state["pos_embed"].shape != self.model.pos_embed.shape:
                raise ValueError(f"Position embedding in {path} does not match {model_type} at 224px.")
            missing, _ = self.model.load_state_dict(state, strict=False)
            if missing:
                raise ValueError(f"Backbone weights in {path} are missing parameters: {missing}")
        self.requires_grad_(trainable)
        self.to(device)
        self.train(trainable)

    def train(self, mode=True):
        return super().train(mode and self.trainable)

    def forward(self, pixel_values):
        pixels = pixel_values.reshape(-1, *pixel_values.shape[-3:])
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            features = self.model(pixels)
        return SimpleNamespace(hidden_states=[features.unsqueeze(1)])
