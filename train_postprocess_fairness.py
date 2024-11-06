import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from evaluate_fairness import evaluate_model, _prepare_dataset_loader, _load_model
from my_models import vit

# Hyperparameters
N_EPOCHS = 15
BATCH_SIZE = 128
IMAGES_LIST_TXT = "work_on_train.txt"

# Initialize the model
model = _load_model(model_path="model_full_train_e12_acc0.887.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare data loaders
train_loader = _prepare_dataset_loader(IMAGES_LIST_TXT)
test_dataset_loader = _prepare_dataset_loader("work_on_test.txt")

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop (without fairness loss)
# for epoch in range(N_EPOCHS):
#     running_loss = 0.0
#     model.train()
#     for i, data in enumerate(train_loader):
#         inputs, labels, race = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs)
#         logits = outputs.logits  # Assuming the model outputs logits
#         loss = criterion(logits, labels)

#         # Backward pass
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 20 == 0:
#             print(
#                 "[Epoch %d, Batch %5d] loss: %.3f"
#                 % (epoch + 1, i + 1, running_loss / 100)
#             )
#             running_loss = 0.0

#     scheduler.step()
#     acc, _ = evaluate_model(model, test_dataset_loader, suppress_printing=True)
#     torch.save(
#         model.state_dict(),
#         f"model_global_threshold_e{epoch + 1}_acc{acc:.3f}.pth",
#     )

# print("Finished Training")

# After training, adjust the global decision threshold
# Step 1: Collect predictions and true labels on the fair set (test dataset with race labels)
model.eval()
y_true = []
y_scores = []
races = []

with torch.no_grad():
    for data in test_dataset_loader:
        inputs, labels, race = data
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Assuming binary classification (class 1 is 'fake')
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())
        races.extend(race)  # Assuming race is a list of strings

y_true = np.array(y_true)
y_scores = np.array(y_scores)
races = np.array(races)

# Step 2: Find the optimal global threshold
def find_optimal_threshold(y_true, y_scores, races):
    thresholds = np.linspace(0, 1, 100)
    min_tpr_diff = float('inf')
    optimal_threshold = 0.5  # Default threshold

    for threshold in thresholds:
        tprs = []
        for race in np.unique(races):
            idx = races == race
            y_true_race = y_true[idx]
            y_pred_race = (y_scores[idx] >= threshold).astype(int)
            tp = np.sum((y_pred_race == 1) & (y_true_race == 1))
            fn = np.sum((y_pred_race == 0) & (y_true_race == 1))
            tpr = tp / (tp + fn + 1e-8)  # Add epsilon to avoid division by zero
            tprs.append(tpr)
        tpr_diff = max(tprs) - min(tprs)
        if tpr_diff < min_tpr_diff:
            min_tpr_diff = tpr_diff
            optimal_threshold = threshold

    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Minimum TPR difference between races: {min_tpr_diff:.4f}")
    return optimal_threshold

optimal_threshold = find_optimal_threshold(y_true, y_scores, races)

# Step 3: Evaluate the model with the new threshold
def evaluate_with_threshold(y_true, y_scores, races, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    accuracy = np.mean(y_pred == y_true)
    print(f"\nAccuracy with threshold {threshold:.4f}: {accuracy:.4f}")

    # Compute TPR and FPR per race
    for race in np.unique(races):
        idx = races == race
        y_true_race = y_true[idx]
        y_pred_race = y_pred[idx]
        tp = np.sum((y_pred_race == 1) & (y_true_race == 1))
        fn = np.sum((y_pred_race == 0) & (y_true_race == 1))
        fp = np.sum((y_pred_race == 1) & (y_true_race == 0))
        tn = np.sum((y_pred_race == 0) & (y_true_race == 0))
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        print(f"Race: {race}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")

# Evaluate with the default threshold (0.5)
print("\nEvaluation with default threshold (0.5):")
evaluate_with_threshold(y_true, y_scores, races, threshold=0.5)

# Evaluate with the optimal threshold
print(f"\nEvaluation with optimal threshold ({optimal_threshold:.4f}):")
evaluate_with_threshold(y_true, y_scores, races, threshold=optimal_threshold)

# During inference, use the optimal threshold
def predict(inputs):
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= optimal_threshold).int()
    return preds

# Save the model and the optimal threshold
torch.save({
    'model_state_dict': model.state_dict(),
    'optimal_threshold': optimal_threshold,
}, 'model_with_optimal_threshold.pth')
