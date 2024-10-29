import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from common import FairDataset, DataFrame
from fairness_metrics import equality_of_odds_parity, predictive_value_parity
from icecream import ic
from evaluate_fairness import evaluate_model

from my_models import tip_learning, vit


# MODEL_PATH = "deepfake_c0_xception.pkl"
N_EPOCHS = 10
BATCH_SIZE = 128
IMAGES_LIST_TXT= "work_on_eq_validate.txt"

model = vit()
model.train()
model = tip_learning(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = transforms.Compose([
    # Resize images to the size expected by the model
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add normalization if required by your model
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# Dataset prepare
train_dataset = FairDataset(
    txt_path=IMAGES_LIST_TXT,
    transformation_function=transform,
    with_predicted=False
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=8
)

# Train parameters prepare
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08
)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)



for epoch in range(N_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # TODO: logits are not for all models
        loss = criterion(outputs.logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 0:
            print("[Epoch %d, Batch %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    scheduler.step()
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}-preprocessed.pth")

print("Finished Training")

evaluate_model(model, train_loader)
