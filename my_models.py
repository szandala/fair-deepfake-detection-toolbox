import torch.nn as nn

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
