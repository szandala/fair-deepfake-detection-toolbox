import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from common import FairDataset, DataFrame, display_parities
from fairness_metrics import equality_of_odds_parity, predictive_value_parity
from icecream import ic


from my_models import vit

MODEL_PATH = "model_epoch_10.pth"
BATCH_SIZE = 128
IMAGES_LIST_TXT= "work_on_test.txt"

def _prepare():
    model = vit()
    model.load_state_dict(torch.load(MODEL_PATH,  weights_only=False))
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
    return model, train_loader

def evaluate_model(model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_expected = []
    all_predicted = []
    all_sensitive_features = []

    with torch.no_grad():
        for i, data in enumerate(train_loader):
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

    # Compute confusion matrices per group
    confusion_matrices = data_frame.confusion_matrix_per_group()

    print("Confusion Matrices per Group for training:")
    print(confusion_matrices)
    print()

    print("Overall Accuracy:")
    print((data_frame.tp()+data_frame.tn)/(data_frame.tp()+data_frame.tn()+data_frame.fn()+data_frame.fp()+))

    tpr_parity, fpr_parity = equality_of_odds_parity(data=data_frame, one=False)
    print("True Positive Parity:")
    display_parities(tpr_parity, text="")
    print("False Positive Parity:")
    display_parities(tpr_parity, text="")
    print()

    ppv_parity, npv_parity = predictive_value_parity(data=data_frame, one=False)
    print("Positive Predictive Value:")
    display_parities(ppv_parity, text="")
    print("Negative Predictive Value:")
    display_parities(npv_parity, text="")


if __name__ == "__main__":
    model, train_loader = _prepare()
    evaluate_model(model, train_loader)
