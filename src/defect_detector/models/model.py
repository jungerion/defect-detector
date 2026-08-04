"""Builds the classifier model — a pretrained ResNet18 with a new final layer."""
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Freeze all the pretrained layers — we don't want to erase what ResNet18
    # already learned about edges/textures/shapes from ImageNet.
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final layer. The original was trained to output 1000 scores
    # (one per ImageNet category) — we replace it with a fresh, UNFROZEN layer
    # that outputs just `num_classes` scores (2, in our case: defective vs ok).
    # This new layer starts randomly initialized and is the ONLY part that
    # actually learns during training.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
