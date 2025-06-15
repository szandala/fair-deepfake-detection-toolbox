import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from .common import FairDataset, DataFrame, display_parities
from .fairness_metrics import equality_of_odds_parity, predictive_value_parity
from icecream import ic
import sys

from .my_models import vit

MODEL_PATH = "model_epoch_10.pth"
BATCH_SIZE = 128
IMAGES_LIST_TXT= "work_on_test.txt"


def _load_model(model_path=MODEL_PATH):
    model = vit()
    model.load_state_dict(torch.load(model_path,  weights_only=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model

def _prepare_dataset_loader(images_list_txt=IMAGES_LIST_TXT, batch_size=BATCH_SIZE, skip_race=None):
    transform = transforms.Compose([
        # Resize images to the size expected by the model
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add normalization if required by your model
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    # Dataset prepare
    test_dataset = FairDataset(
        txt_path=images_list_txt,
        transformation_function=transform,
        with_predicted=False,
        skip_race=skip_race
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8
    )
    return test_loader

def evaluate_model(model, test_loader, suppres_printing=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_expected = []
    all_predicted = []
    all_sensitive_features = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, race = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)

            all_expected.extend(labels.numpy())
            all_predicted.extend(preds.cpu().numpy())
            all_sensitive_features.extend(race)

            if i%10 == 0:
                print(f"Processed {i*BATCH_SIZE} images")

    # Create a DataFrame for analysis
    data_frame = DataFrame(
        predicted=all_predicted,
        expected=all_expected,
        groups=all_sensitive_features
    )

    acc = (data_frame.tp()+data_frame.tn())/(data_frame.tp()+data_frame.tn()+data_frame.fn()+data_frame.fp())
    f1 = (2*data_frame.tp())/(2*data_frame.tp()+data_frame.fn()+data_frame.fp())
    if suppres_printing:
        return acc, data_frame
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Overall F-1 score: {f1:.4f}")
    print("Confusion Matrices per Group for training:")
    confusion_matrices = data_frame.confusion_matrix_per_group()
    print(confusion_matrices)
    print()



    tpr_parity, fpr_parity = equality_of_odds_parity(data=data_frame, one=True)
    print("True Positive Parity:")
    display_parities(tpr_parity, text="")
    print("False Positive Parity:")
    display_parities(fpr_parity, text="")
    print()

    ppv_parity, npv_parity = predictive_value_parity(data=data_frame, one=True)
    print("Positive Predictive Value:")
    display_parities(ppv_parity, text="")
    print("Negative Predictive Value:")
    display_parities(npv_parity, text="")
    return acc, data_frame



def evaluate_model_logits(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_logits = []
    all_expected = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, _ = data  # Assuming 'race' is not needed here
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            logits = outputs.logits  # Extract the logits tensor
            probs = torch.softmax(logits, dim=1)[:, 1]  # Assuming binary classification (class 1 is 'fake')
            # Move logits and labels to CPU and convert to numpy arrays
            logits = probs.cpu().numpy()
            labels = labels.cpu().numpy()

            # For each sample in the batch, collect the logits and labels
            for logit, label in zip(logits, labels):
                all_logits.append(logit)
                all_expected.append(label)

            if i % 10 == 0:
                print(f"Processed {i * test_loader.batch_size} images")

    # Return the list of tuples (logits, expected_label)
    return list(zip(all_logits, all_expected))

def draw_histogram(results):
    import matplotlib.pyplot as plt
    import numpy as np

    logit_diffs, expected_labels = zip(*results)

    logit_diffs = np.array(logit_diffs)
    expected_labels = np.array(expected_labels)
    # mask = (logit_diffs > 0.01) & (logit_diffs < 0.99)
    mask = True
    logit_diffs_m = logit_diffs[mask]
    expected_labels_m = expected_labels[mask]

    plt.hist(logit_diffs_m[expected_labels_m == 0], bins=100, color='blue', alpha=1, label='Class 0')
    plt.hist(logit_diffs_m[expected_labels_m == 1], bins=100, color='red', alpha=1, label='Class 1')
    plt.xlabel('Logit Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Logit Differences by Class')
    plt.legend()
    plt.savefig('logit_differences_histogram.png', dpi=300)



if __name__ == "__main__":
    model_path = sys.argv[1]
    ic(model_path)
    model = _load_model(model_path)
    test_loader = _prepare_dataset_loader()
    # evaluate_model(model, test_loader)
    results = evaluate_model_logits(model, test_loader)
    draw_histogram(results)
