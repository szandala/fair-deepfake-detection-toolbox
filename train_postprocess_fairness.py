import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from evaluate_fairness import evaluate_model, _prepare_dataset_loader, _load_model
from my_models import vit

# Hyperparameters
MODEL_PATH="model_full_undersampl_train_e14_acc0.756.pth"
N_EPOCHS = 15
BATCH_SIZE = 128
IMAGES_LIST_TXT = "work_on_train.txt"

# Initialize the model
model = _load_model(model_path=MODEL_PATH)
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
        ppvs = []
        npvs = []
        for race in np.unique(races):
            idx = races == race
            y_true_race = y_true[idx]
            y_pred_race = (y_scores[idx] >= threshold).astype(int)

            tp = np.sum((y_pred_race == 1) & (y_true_race == 1))
            fn = np.sum((y_pred_race == 0) & (y_true_race == 1))
            fp = np.sum((y_pred_race == 1) & (y_true_race == 0))
            tn = np.sum((y_pred_race == 0) & (y_true_race == 0))

            tpr = tp / (tp + fn)   # True Positive Rate
            fpr = fp / (fp + tn)   # False Positive Rate
            tprs.append(tpr)
            fprs.append(fpr)
            ppv = tp/(tp+fp)
            npv = tn/(tn+fn)
            ppvs.append(ppv)
            npvs.append(npv)


        # Calculate the ratios
        tpr_ratio = min(tprs)/max(tprs)
        fpr_ratio = (min(fprs)/max(fprs)) * (1 - max(fprs)) # FPRs aim to 0, so adding extra penalty
        ppv_ratio = min(ppvs)/max(ppvs)
        npv_ratio = min(npvs)/max(npvs)

        # Combine ratios
        total_ratio = tpr_ratio **2 * fpr_ratio **2 * ppv_ratio **2 * npv_ratio**2
        # ic(total_ratio)
        if total_ratio > max_total_ratio:
            max_total_ratio = total_ratio
            optimal_threshold = threshold

    print(f"Optimal threshold: {optimal_threshold:.4f}")
    # print(f"Minimum total difference (TPR diff + FPR diff) between races: {max_total_ratio:.4f}")
    return optimal_threshold

optimal_threshold = find_optimal_threshold(y_true, y_scores, races)


# Step 3: Adjust the model's bias term to bake the threshold into the model

# Calculate the bias adjustment
delta_b = -np.log((1 - optimal_threshold) / optimal_threshold)

print(f"Bias adjustment (delta_b): {delta_b:.4f}")

# Adjust the model's bias term
# Assuming the last layer is a Linear layer named 'classifier' (common in models like ViT)

# Access the last Linear layer
last_linear = model.classifier

# Check if the last layer is indeed Linear
if not isinstance(last_linear, nn.Linear):
    raise TypeError("The last layer is not a Linear layer. Please adjust the code accordingly.")

# Adjust the bias term
with torch.no_grad():
    # Get the weights and biases
    weights = last_linear.weight  # Shape: [num_classes, in_features]
    biases = last_linear.bias     # Shape: [num_classes]

    # Adjust the biases
    # For binary classification, adjust the biases of both classes accordingly
    # Adjust the bias of class 1 (index 1)
    biases_adjusted = biases.clone()
    biases_adjusted[1] += delta_b
    # Update the bias
    last_linear.bias.copy_(biases_adjusted)

print("Bias term adjusted to bake in the optimal threshold.")



# Step 4: Evaluate the model with the new threshold
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
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        ppv = tp /(tp+fp)
        npv = tn/(tn+fn)
        print(f"Race: {race}, TPR: {tpr:.4f}, FPR: {fpr:.4f}, PPV: {ppv:.4f}, NPV: {npv:.4f}")
    return accuracy

# Evaluate with the default threshold (0.5)
print("\nEvaluation with default threshold (0.5):")
evaluate_with_threshold(y_true, y_scores, races, threshold=0.5)

# Evaluate with the optimal threshold
print(f"\nEvaluation with optimal threshold ({optimal_threshold:.4f}):")
acc = evaluate_with_threshold(y_true, y_scores, races, threshold=optimal_threshold)

# During inference, use the optimal threshold
def predict(inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.to(device))
        logits = outputs.logits  # Extract the logits tensor
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= optimal_threshold).int()
    return preds
new_model_path = f"{MODEL_PATH.replace(".pth", f"_thrt")}_acc{acc:.4f}.pth"
# Save the model and the optimal threshold
torch.save(model.state_dict(), new_model_path)
