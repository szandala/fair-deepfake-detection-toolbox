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
model = _load_model(model_path="model_full_undersampl_train_e14_acc0.756.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare data loader
test_dataset_loader = _prepare_dataset_loader("work_on_undersampl_train.txt")

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
        outputs = model(inputs)
        logits = outputs.logits  # Extract the logits tensor
        probs = torch.softmax(logits, dim=1)[:, 1]  # Assuming binary classification (class 1 is 'fake')
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())
        races.extend(race)  # Assuming race is a list of strings

y_true = np.array(y_true)
y_scores = np.array(y_scores)
races = np.array(races)

# Step 2: Find the optimal global threshold
def find_optimal_threshold(y_true, y_scores, races):
    thresholds = np.linspace(0, 1, 10000)
    max_total_ratio = -1
    optimal_threshold = 0.5  # Default threshold

    for threshold in thresholds:
        tprs = []
        fprs = []
        for race in np.unique(races):
            idx = races == race
            y_true_race = y_true[idx]
            y_pred_race = (y_scores[idx] >= threshold).astype(int)

            tp = np.sum((y_pred_race == 1) & (y_true_race == 1))
            fn = np.sum((y_pred_race == 0) & (y_true_race == 1))
            fp = np.sum((y_pred_race == 1) & (y_true_race == 0))
            tn = np.sum((y_pred_race == 0) & (y_true_race == 0))

            # tpr = tp / (tp + fn) if (tp + fn) != 0 else 1  # True Positive Rate
            # fpr = fp / (fp + tn) if (fp + tn) != 0 else 1  # False Positive Rate
            ppv = tp/(tp+fp)
            npv = tn/(tn+fn)
            tprs.append(ppv)
            fprs.append(npv)

        # Calculate the ratios
        tpr_ratio = min(tprs)/max(tprs)
        fpr_ratio = min(fprs)/max(fprs)

        # Combine TPR and FPR ratios
        total_ratio = tpr_ratio + fpr_ratio  # You can adjust this combination as needed
        # ic(total_ratio)
        if total_ratio > max_total_ratio:
            max_total_ratio = total_ratio
            optimal_threshold = threshold

    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Minimum total difference (TPR diff + FPR diff) between races: {max_total_ratio:.4f}")
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
        outputs = model(inputs.to(device))
        logits = outputs.logits  # Extract the logits tensor
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= optimal_threshold).int()
    return preds

# Save the model and the optimal threshold
torch.save({
    'model_state_dict': model.state_dict(),
    'optimal_threshold': optimal_threshold,
}, 'model_with_optimal_threshold.pth')
