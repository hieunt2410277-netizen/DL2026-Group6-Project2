import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes=200, strategy="frozen"):
    """
    strategy:
      - frozen: freeze the pretrained backbone and train only the classifier.
      - finetune: train the whole pretrained model.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if strategy == "frozen":
        for param in model.parameters():
            param.requires_grad = False
    elif strategy != "finetune":
        raise ValueError("strategy must be 'frozen' or 'finetune'")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
