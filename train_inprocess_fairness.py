import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from libs.fairness_metrics import equality_of_odds_parity, predictive_value_parity
from icecream import ic
from libs.evaluate_fairness import evaluate_model, _prepare_dataset_loader

from libs.my_models import tip_learning, vit


N_EPOCHS = 15
BATCH_SIZE = 128
IMAGES_LIST_TXT = "work_on_train.txt"
SKIP_RACE="asian"

model = vit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def compute_fairness_loss(labels, preds, sensitive_features):
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    # sensitive_features_np = sensitive_features

    tpr_parity, fpr_parity = equality_of_odds_parity(
        expected=labels_np,
        predicted=preds_np,
        sensitive_features=sensitive_features,
        one=True,
    )
    ppv_parity, npv_parity = predictive_value_parity(
        expected=labels_np,
        predicted=preds_np,
        sensitive_features=sensitive_features,
        one=True,
    )
    # FPR should go to 0.0,
    # so overall, in perfect world, it would be 0/0 to get min/max
    # this "trick" solves this concern.
    # fpr_parity = 1 if fpr_parity == 0 else fpr_parity

    # Fairness loss is the std deviation from perfect parity (which is 1)
    fairness_loss = (1 - tpr_parity) ** 2 + (1 - fpr_parity) ** 2 + (1- ppv_parity) ** 2 + (1- npv_parity) ** 2
    fairness_loss = torch.tensor(fairness_loss, device=labels.device, dtype=torch.float)

    return fairness_loss


train_loader = _prepare_dataset_loader(IMAGES_LIST_TXT, skip_race=SKIP_RACE)
test_dataset_loader = _prepare_dataset_loader("work_on_test.txt")

# Train parameters prepare
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


for epoch in range(N_EPOCHS):
    running_loss = 0.0
    model.train()  # in case I change in eval into model.eval()
    # model = tip_learning(model)
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
            print(
                "[Epoch %d, Batch %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / 100)
            )
            running_loss = 0.0

    scheduler.step()
    acc, _ = evaluate_model(model, test_dataset_loader, suppres_printing=False)
    torch.save(
        model.state_dict(),
        f"model_full_inprocess_4metr_train_no-{SKIP_RACE}_e{epoch + 1}_acc{acc:.3f}.pth",
    )

print("Finished Training")

# evaluate_model(model, train_loader)
