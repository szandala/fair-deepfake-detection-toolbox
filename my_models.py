import torch.nn as nn
from torchvision import models

from transformers import ViTForImageClassification

def tip_learning(model):
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model

def vit():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

def efficientnet_b4():
    # Load the pretrained EfficientNet-B4 model
    model = models.efficientnet_b4(pretrained=True)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    return model

def resnet152():
    model = models.resnet152(pretrained=True)

    # Modify the fully connected layer to output 2 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
