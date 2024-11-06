import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from common import FairDataset, DataFrame
from fairness_metrics import equality_of_odds_parity, predictive_value_parity
from icecream import ic
from evaluate_fairness import evaluate_model, _prepare_dataset_loader

from my_models import tip_learning, vit, efficientnet_b4, resnet101


# MODEL_PATH = "deepfake_c0_xception.pkl"
N_EPOCHS = 15
BATCH_SIZE = 64
IMAGES_LIST_TXT= "work_on_train.txt"

model = resnet101()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_loader = _prepare_dataset_loader(IMAGES_LIST_TXT)
test_dataset_loader = _prepare_dataset_loader("work_on_test.txt")

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
    model.train() # in case I change in eval into model.eval()
    # model = tip_learning(model)
    for i, data in enumerate(train_loader):
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # TODO: logits are not for all models, e.g. resnet
        # loss = criterion(outputs.logits, labels)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 0:
            print("[Epoch %d, Batch %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    scheduler.step()
    acc, _ = evaluate_model(model, test_dataset_loader, suppres_printing=True)
    torch.save(model.state_dict(), f"model_res101_full_train_e{epoch + 1}_acc{acc:.3f}.pth")

print("Finished Training")
