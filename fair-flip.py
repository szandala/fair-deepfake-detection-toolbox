import torch
import torch.nn as nn
from evaluate_fairness import evaluate_model, _prepare_dataset_loader, _load_model

import numpy as np
import sys

# --------------------------------------------------------------
# 1. Hook, for collecting input to LAST LAYER
# --------------------------------------------------------------
feature_storage = {
    "x_input_list": [],
    "labels_list": [],
    "races_list": [],
    "preds_list": [],
}
SKIP_RACE = "black"

def classifier_input_hook(module, input_, output):
    """
    Hook is called on each batch that reached final layer
    """
    x_input = input_[0].detach().cpu()  # shape: (batch_size, in_features)
    feature_storage["x_input_list"].append(x_input)


# --------------------------------------------------------------
# 2. Computing STD for each race, for each class
#    Assumptions:
#      - preds = predictions -> {0,1}
#      - races protected attributes  -> {"White", "Black", ...}
#      - X_input input to final layer)
#
#   For each class (0,1), for each ethnicity we compute STD of each FINAL LAYER INPUT
#   Then we compute average value of STD for each input for each class
#   ... yes, I know, complicated...
# --------------------------------------------------------------
def compute_std_across_races(preds, races, X_input):
    """
    Returns:
      std_class0, std_class1 -> tensor of shape (in_features,)
      where each vlaue is an avarage STD across all ethnicities
    """
    unique_races = np.unique(races)
    in_features = X_input.shape[1]

    X_class0 = []
    R_class0 = []
    X_class1 = []
    R_class1 = []

    for i in range(len(preds)):
        if preds[i] == 0:
            X_class0.append(X_input[i])
            R_class0.append(races[i])
        else:
            X_class1.append(X_input[i])
            R_class1.append(races[i])

    X_class0 = np.array(X_class0)  # (N0, in_features)
    R_class0 = np.array(R_class0)
    X_class1 = np.array(X_class1)  # (N1, in_features)
    R_class1 = np.array(R_class1)

    # We compute STD for each race

    std_accum_0 = []
    for r in unique_races:
        idx_r = R_class0 == r
        if np.sum(idx_r) > 1:
            std_r = np.std(X_class0[idx_r], axis=0)  # shape: (in_features,)
            std_accum_0.append(std_r)
        else:
            # no results for given class -> zeros
            std_accum_0.append(np.zeros(in_features, dtype=np.float32))
    # We compute mean across all STDs
    std_class0 = (
        np.mean(std_accum_0, axis=0) if len(std_accum_0) > 0 else np.zeros(in_features)
    )

    std_accum_1 = []
    for r in unique_races:
        idx_r = R_class1 == r
        if np.sum(idx_r) > 1:
            std_r = np.std(X_class1[idx_r], axis=0)
            std_accum_1.append(std_r)
        else:
            std_accum_1.append(np.zeros(in_features, dtype=np.float32))

    std_class1 = (
        np.mean(std_accum_1, axis=0) if len(std_accum_1) > 0 else np.zeros(in_features)
    )
    return torch.from_numpy(std_class0), torch.from_numpy(std_class1)


# --------------------------------------------------------------
# 3. Final layer weight scaling
# --------------------------------------------------------------
def scale_last_layer_weights_multiply(model, std_class0, std_class1, alpha):
    """
    Modify weights `model.classifier` (nn.Linear):
      - Low deviating inputs are enhances (scale > 1)
      - High deviating inputs are dampen (scale < 1)
    Range of scaling: [1-alpha, 1+alpha].

    Args:
      model: model with model.classifier = nn.Linear(in_features, 2)
      std_class0, std_class1: (in_features,),
        Average STDs screed protectd groups
      alpha: scaling factor
            (eg. alpha=0.2 => 0.8 do 1.2)
    """

    if not isinstance(model.classifier, nn.Linear):
        raise TypeError("Final layer is NOT nn.Linear!")

    W = model.classifier.weight.data.clone()  # (2, in_features)
    b = model.classifier.bias.data.clone()  # (2,)

    in_features = W.shape[1]

    # 1) Normalise deviations to have them in range [0,1]
    all_std = torch.cat([std_class0, std_class1], dim=0)  # (2*in_features,)
    std_min = all_std.min()
    std_max = all_std.max()
    eps = 1e-8

    #    (low STD -> 0, high STD -> 1)
    std_class0_norm = (std_class0 - std_min) / (std_max - std_min + eps)
    std_class1_norm = (std_class1 - std_min) / (std_max - std_min + eps)

    # 2) Scale function:
    #    TODO: this is field for changes
    def scale_fn(std_norm):
        # return (1.0 + alpha) - 2.0 * alpha * std_norm
        return 1 + alpha - std_norm * 2

    # 3) Scale input weights by factor in range [0,1]
    for n in range(in_features):
        # class 0
        old_w0 = W[0, n]
        s0 = scale_fn(std_class0_norm[n])
        W[0, n] = old_w0 * s0
        # print(f"Change: {old_w0} -> {old_w0 * s0} (by *{s0})")

        # class 1
        old_w1 = W[1, n]
        s1 = scale_fn(std_class1_norm[n])
        W[1, n] = old_w1 * s1
        # print(f"Change: {old_w1} -> {old_w1 * s1} (by *{s1})")

    # 4) Apply chnges to the model
    model.classifier.weight.data.copy_(W)
    model.classifier.bias.data.copy_(b)

    print(
        f">> [Multiply-SCALE] Weights updated with alpha={alpha}. "
        "Stable dims got >1 scale, high-variance dims got <1 scale."
    )


if __name__ == "__main__":

    alpha = float(sys.argv[1])
    model_path = float(sys.argv[2])
    # 1) Defaults
    MODEL_PATH = model_path
    # "model_full_train_e12_acc0.887.pth"
    BATCH_SIZE = 32
    IMAGES_LIST_TXT = "work_on_train.txt"
    TEST_LIST_TXT = "work_on_test.txt"

    # 2) Load model
    model = _load_model(model_path=MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 3) Apply HOOK
    hook_handle = model.classifier.register_forward_hook(classifier_input_hook)

    # 4) Load Data
    train_loader = _prepare_dataset_loader(IMAGES_LIST_TXT, batch_size=BATCH_SIZE, SKIP_RACE=SKIP_RACE)
    test_loader = _prepare_dataset_loader(TEST_LIST_TXT, batch_size=BATCH_SIZE)

    # 5) Collect model's behaviour
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if i % 5000 == 1:
                print(f"{i-1} done")
            inputs, labels, races = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            # preds -> argmax(logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            feature_storage["preds_list"].extend(preds.tolist())
            feature_storage["races_list"].extend(races)
            feature_storage["labels_list"].extend(labels.numpy().tolist())

    x_input_cat = torch.cat(feature_storage["x_input_list"], dim=0)  # (N, in_features)
    preds_np = np.array(feature_storage["preds_list"])  # (N,)
    races_np = np.array(feature_storage["races_list"])  # (N,)
    labels_np = np.array(feature_storage["labels_list"])  # (N,)

    # 6) Compute STDs
    std_class0, std_class1 = compute_std_across_races(
        preds_np, races_np, x_input_cat.numpy()
    )

    # 7) Scale weights
    scale_last_layer_weights_multiply(model, std_class0, std_class1, alpha=alpha)

    # 8) Disable hook - not needed anymore
    hook_handle.remove()

    # 9) Evluate model
    acc_after, _ = evaluate_model(model, test_loader, suppres_printing=False)
    print(f"Accuracy after weight scaling: {acc_after:.3f}")

    # 10) Save new model
    torch.save(model.state_dict(), f"fair-flip_models/fair-flip_no-{SKIP_RACE}_a{acc_after:.3f}_alpha{alpha}.pth")

    print("Done, I'm exciting :D")
