import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from evaluate_fairness import evaluate_model, _prepare_dataset_loader, _load_model
from common import FairDataset

# --------------------------------------------------------------
# BPFA – Dane pomocnicze przechwytywane w HOOK
# --------------------------------------------------------------
feature_storage = {
    "x_input_list": [],  # wektory wejściowe do classifiera
    "labels_list": [],
    "races_list": [],
    "preds_list": []
}
conv_outputs_storage = {}

current_batch_races = []  # We'll set this before model(inputs)

def conv_forward_hook_factory(layer_id):
    """
    Creates a hook function that captures the output (feature maps) of this conv layer,
    grouping them by race in conv_outputs_storage[layer_id]["race_to_outputs"].
    """
    def hook_fn(module, input_, output):
        global current_batch_races
        # output: shape (B, C_out, H_out, W_out)
        out_cpu = output.detach().cpu()
        for i, r in enumerate(current_batch_races):
            race_dict = conv_outputs_storage[layer_id]["race_to_outputs"]
            if r not in race_dict:
                race_dict[r] = []
            # shape for this single sample: (C_out, H_out, W_out)
            race_dict[r].append(out_cpu[i].numpy())
    return hook_fn

def register_all_conv_hooks(model, prefix=""):
    """
    Recursively travels the model, and for each Conv2d, registers a forward hook.
    prefix is a string to identify the layer uniquely, e.g. 'block1.conv2'
    """
    for name, child in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Conv2d):
            # Prepare storage for this layer
            conv_outputs_storage[layer_name] = {"race_to_outputs": {}}

            # Register the hook
            hook_fn = conv_forward_hook_factory(layer_name)
            child.register_forward_hook(hook_fn)
            print(f"Registered hook on Conv2d: {layer_name}")

        else:
            # Recursively process submodules
            register_all_conv_hooks(child, prefix=layer_name)

# --------------------------------------------------------------
# BPFA – Funkcje obliczające "bias" i wykonujące pruning
# --------------------------------------------------------------
def compute_bias_per_input_dim(x_input_cat: torch.Tensor,
                               races_np: np.ndarray) -> np.ndarray:
    """
    Dla każdej kolumny (wymiaru wejściowego) oblicza wartość BIAS_j = std(Z^s_j),
    gdzie
       Z^s_j = srednia z |x_input_cat[probek z rasy s, j]|.
    Zwraca tablicę biasów o rozmiarze (in_features,).
    """
    x_input_np = x_input_cat.numpy()  # (N, in_features)
    unique_races = np.unique(races_np)

    in_features = x_input_np.shape[1]
    bias_array = np.zeros(in_features, dtype=np.float32)

    # Dla każdej kolumny obliczamy Z^s_j, a potem std po s
    for j in range(in_features):
        # Zbierzemy dla każdej rasy s -> uśrednimy moduł
        z_s_values = []
        for s in unique_races:
            # Indeksy próbek, które należą do rasy s
            idx_s = (races_np == s)
            # Weź wartości w kolumnie j
            x_s = x_input_np[idx_s, j]
            if len(x_s) > 0:
                z_s = np.mean(np.abs(x_s))
            else:
                z_s = 0.0
            z_s_values.append(z_s)
        # std po rasach
        bias_j = np.std(z_s_values)
        bias_array[j] = bias_j

    return bias_array

def bpfa_prune_conv_layer(conv_layer: nn.Conv2d,
                          bias_per_filter: np.ndarray,
                          pruning_rate: float = 0.2):
    """
    Prune the lowest (pruning_rate*100)% of weights in conv_layer by:
       PS_{i,j,k,m} = abs(W_{i,j,k,m}) / bias_per_filter[i]
    i: filter index (output channel)
    j: input channel
    k,m: kernel coords

    Then set those with the smallest PS to zero.
    """
    W = conv_layer.weight.data  # shape: (C_out, C_in, K_h, K_w)
    C_out, C_in, K_h, K_w = W.shape

    # Flatten for scoring
    W_cpu = W.detach().cpu().numpy().reshape(C_out, C_in*K_h*K_w)

    # We'll collect (score, i, w_idx) in a list
    scores_list = []
    for i in range(C_out):
        b = bias_per_filter[i]
        # if b is 0 or near 0 => treat it carefully (like stable => big denominator => small PS => we won't prune?)
        if b < 1e-8:
            b = 1e-8

        for w_idx in range(C_in*K_h*K_w):
            w_val = W_cpu[i, w_idx]
            ps = abs(w_val)/b
            scores_list.append((ps, i, w_idx))

    # Sort ascending
    scores_list.sort(key=lambda x: x[0])
    total = len(scores_list)
    num_to_prune = int(pruning_rate*total)

    if num_to_prune == 0:
        print(f"No weights pruned in this layer (pruning_rate={pruning_rate} was too small).")
        return

    threshold = scores_list[num_to_prune][0]
    print(f"Layer {conv_layer} => total={total}, prune={num_to_prune}, threshold={threshold:.6f}")

    # Actually zero out
    for idx in range(num_to_prune):
        _, fil_i, w_idx = scores_list[idx]
        c_in = w_idx // (K_h*K_w)
        rest = w_idx % (K_h*K_w)
        k_h = rest // K_w
        k_w = rest % K_w

        W[fil_i, c_in, k_h, k_w] = 0.0

    print(f"Pruning done for layer {conv_layer}. Lowest scores set to zero.")
def compute_bias_per_filter_for_layer(layer_name):
    """
    For the given layer_name, we have:
      conv_outputs_storage[layer_name]["race_to_outputs"] -> { race: [np_array(Cout,Hout,Wout), ...], ... }

    We'll compute the "bias" for each filter i in that layer, i.e. the across-race std of the average L2 norm.
      BIAS_i = std_races( mean_over_samples_in_race( L2norm_of_filter_i ) ).
    Returns: np.array of shape (C_out,)
    """
    race_to_outs = conv_outputs_storage[layer_name]["race_to_outputs"]
    races = list(race_to_outs.keys())

    # Let's check shape from any sample
    # pick the first race with at least 1 sample
    any_race = None
    for r in races:
        if len(race_to_outs[r]) > 0:
            any_race = r
            break
    if not any_race:
        # no data? return empty
        return None

    sample0 = race_to_outs[any_race][0]  # shape (C_out, H_out, W_out)
    C_out = sample0.shape[0]

    # We'll store "average L2 norms" per race -> shape (C_out,)
    racewise_means = []
    for r in races:
        arrs = race_to_outs[r]  # list of np arrays
        if len(arrs) == 0:
            # no samples for this race
            racewise_means.append(np.zeros(C_out, dtype=np.float32))
            continue

        # Accumulate L2 norms for each filter
        all_l2 = []
        for fm in arrs:
            # fm shape: (C_out, H_out, W_out)
            # L2 for each filter i
            # sum squares over H_out, W_out
            fm_sq = fm * fm
            sum_hw = fm_sq.reshape(C_out, -1).sum(axis=1)  # (C_out,)
            l2_each_filter = np.sqrt(sum_hw)               # (C_out,)
            all_l2.append(l2_each_filter)

        # (num_samples_race, C_out)
        stacked = np.vstack(all_l2)
        # average across samples of that race => shape (C_out,)
        racewise_mean = stacked.mean(axis=0)
        racewise_means.append(racewise_mean)

    # shape: (num_races, C_out)
    racewise_matrix = np.vstack(racewise_means)
    # Now BIAS_i = std over races for filter i
    bias_per_filter = racewise_matrix.std(axis=0)  # shape (C_out,)
    return bias_per_filter

# --------------------------------------------------------------
# Przykładowy SKRYPT główny z zaimplementowanym BPFA
# --------------------------------------------------------------
def prune_all_conv_layers_bpfa(model, pruning_rate=0.2):
    """
    For each conv layer in the model:
      1) compute bias_per_filter
      2) do unstructured pruning
    """
    for layer_id, layer_dict in conv_outputs_storage.items():
        # layer_id is a string like "layer1.0.conv2" or "features.3"
        race_dict = layer_dict["race_to_outputs"]
        # If no data was collected, skip
        if not race_dict:
            continue
        # We need to find the actual conv layer object
        # We can do that by a small helper that finds the layer by name
        conv_layer = get_layer_by_name(model, layer_id)
        if not isinstance(conv_layer, nn.Conv2d):
            continue

        # Compute bias
        bias_filter = compute_bias_per_filter_for_layer(layer_id)  # shape: (C_out,)
        if bias_filter is None:
            print(f"No data for layer {layer_id}, skipping")
            continue

        # Prune
        bpfa_prune_conv_layer(conv_layer, bias_filter, pruning_rate=pruning_rate)

def get_layer_by_name(model, layer_name):
    """
    A small helper that traverses model.named_modules()
    and returns the module whose name == layer_name.
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


# Example main script
if __name__ == "__main__":
    # 1) Load your model
    MODEL_PATH = "model_full_train_e12_acc0.887.pth"
    IMAGES_LIST_TXT = "work_on_train.txt"
    TEST_LIST_TXT = "work_on_test.txt"
    BATCH_SIZE = 32

    model = _load_model(MODEL_PATH)
    model.eval()
    model.to("cuda")

    # 2) Register hooks on all conv layers
    register_all_conv_hooks(model)

    # 3) Data loader
    train_loader = _prepare_dataset_loader(IMAGES_LIST_TXT, batch_size=BATCH_SIZE)
    test_loader = _prepare_dataset_loader(TEST_LIST_TXT, batch_size=BATCH_SIZE)

    # Evaluate pre-pruning
    acc_before, _ = evaluate_model(model, test_loader)
    print(f"Accuracy before: {acc_before:.3f}")

    # 4) Inference pass, capturing conv outputs
    # We'll do it on test set or the entire dataset, your choice
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, races = data
            inputs = inputs.to("cuda")

            # set global current_batch_races so the hook can read them
            current_batch_races = races

            _ = model(inputs)  # triggers hooks

    # 5) Now we have conv_outputs_storage filled with all layer outputs by race
    #    Prune each conv layer
    # 0.04 seems the best according to the authors
    prune_all_conv_layers_bpfa(model, pruning_rate=0.04)

    # 6) Evaluate post-pruning
    acc_after, _ = evaluate_model(model, test_loader)
    print(f"Accuracy after pruning: {acc_after:.3f}")

    # 7) Optionally save
    torch.save(model.state_dict(), "my_model_after_bpfa_all_convs.pth")
    print("Done.")
