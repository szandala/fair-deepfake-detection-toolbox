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


MODEL_PATH = "model_epoch_2.pth"
N_EPOCHS = 2
BATCH_SIZE = 128
IMAGES_LIST_TXT= "work_on_validate.txt"

model = vit()
model.load_state_dict(torch.load(MODEL_PATH))
model.train()
model = tip_learning(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# def compute_fairness_loss(labels, preds, sensitive_attrs):
#     unique_groups = torch.unique(sensitive_attrs)
#     tpr_values = []
#     fpr_values = []

#     for group in unique_groups:
#         group_mask = (sensitive_attrs == group)
#         group_labels = labels[group_mask]
#         group_preds = preds[group_mask]

#         TP = ((group_labels == 1) & (group_preds == 1)).sum()
#         FP = ((group_labels == 0) & (group_preds == 1)).sum()
#         TN = ((group_labels == 0) & (group_preds == 0)).sum()
#         FN = ((group_labels == 1) & (group_preds == 0)).sum()

#         if (TP + FN) > 0:
#             TPR = TP.float() / (TP + FN).float()
#         else:
#             TPR = torch.tensor(0.0, device=TP.device)

#         if (FP + TN) > 0:
#             FPR = FP.float() / (FP + TN).float()
#         else:
#             FPR = torch.tensor(0.0, device=FP.device)

#         tpr_values.append(TPR)
#         fpr_values.append(FPR)

#     tpr_values = torch.stack(tpr_values)
#     fpr_values = torch.stack(fpr_values)

#     tpr_var = torch.var(tpr_values, unbiased=False)
#     fpr_var = torch.var(fpr_values, unbiased=False)

#     fairness_loss = tpr_var + fpr_var
#     return fairness_loss

def compute_fairness_loss(labels, preds, sensitive_features):
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    # sensitive_features_np = sensitive_features

    tpr_parity, fpr_parity = equality_of_odds_parity(
        expected=labels_np,
        predicted=preds_np,
        sensitive_features=sensitive_features,
        one=True
    )

    # Fairness loss is the deviation from perfect parity (which is 1)
    fairness_loss = (1 - tpr_parity) ** 2 + (1 - fpr_parity) ** 2
    fairness_loss = torch.tensor(fairness_loss, device=labels.device, dtype=torch.float)

    return fairness_loss



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
        inputs, labels, race = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # TODO: logits are not for all models
        loss = criterion(outputs.logits, labels)
        _, preds = torch.max(outputs.logits, 1)
        fairness_loss = compute_fairness_loss(labels, preds, race)

        # Backward pass
        total_loss = loss + fairness_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        if i % 20 == 0:
            print("[Epoch %d, Batch %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    scheduler.step()
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}-inprocessed.pth")

print("Finished Training")

evaluate_model(model, train_loader)
