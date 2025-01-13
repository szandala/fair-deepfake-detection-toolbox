import torch
import torch.nn as nn
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

def classifier_input_hook(module, input_, output):
    """
    Funkcja hook wywoływana ZA KAŻDYM RAZEM, gdy batch przechodzi przez
    `model.classifier(...)`.
    Zapisujemy TYLKO input do globalnej listy, żeby potem obliczyć bias wg rasy.
    """
    x_input = input_[0].detach().cpu()  # shape: (batch_size, in_features)
    feature_storage["x_input_list"].append(x_input)

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


def bpfa_pruning_linear_layer(linear_layer: nn.Linear,
                              bias_per_dim: np.ndarray,
                              pruning_rate: float = 0.2):
    """
    Wykonuje pruning wg opisanego algorytmu:
      PS_{i,j} = W_{i,j} / BIAS_j
    i zeruje (w prymitywny sposób) pewien odsetek wag o najmniejszej wartości |PS|.

    - linear_layer.weight shape: (out_features, in_features)
    - bias_per_dim shape: (in_features,)
    """
    W = linear_layer.weight.data  # (out_features, in_features)
    out_features, in_features = W.shape

    # Obliczamy macierz PS o kształcie (out_features, in_features)
    # Uwaga: jeśli BIAS_j=0, trzeba uważać, by nie dzielić przez zero.
    ps = torch.zeros_like(W)
    for j in range(in_features):
        if bias_per_dim[j] != 0.0:
            ps[:, j] = W[:, j] / bias_per_dim[j]
            print("in IF")
        else:
            # Jeśli bias == 0, to "teoretycznie" w ogóle nie mamy zróżnicowania między rasami,
            # więc można np. pominąć wagi z tej kolumny bądź nie usuwać ich wcale.
            ps[:, j] = W[:, j]
            print("in ELSE")

    # Flattenujemy i wybieramy próg do ścięcia
    ps_abs = ps.abs().view(-1)
    num_weights = ps_abs.shape[0]
    k = int(pruning_rate * num_weights)  # ile wag zerujemy

    print(f"Pruning {k=}")
    if k == 0:
        return  # nic nie wycinamy, bo pruning_rate zbyt mały

    # Znajdź wartość progu (k-ta najmniejsza wartość w |PS|)
    threshold = torch.kthvalue(ps_abs, k).values.item()

    # Wyzeruj w oryginalnych wagach te elementy, których |PS| <= threshold
    mask = (ps.abs() > threshold)
    W *= mask  # in-place

# --------------------------------------------------------------
# Przykładowy SKRYPT główny z zaimplementowanym BPFA
# --------------------------------------------------------------
if __name__ == "__main__":

    # ---------------- 1) Parametry ----------------
    MODEL_PATH = "model_full_train_e12_acc0.887.pth"
    N_EPOCHS = 2
    BATCH_SIZE = 32
    IMAGES_LIST_TXT = "work_on_train.txt"
    TEST_LIST_TXT = "work_on_test.txt"
    PRUNING_RATE = 0.2  # np. 20% wag do wycięcia

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- 2) Wczytujemy model ViT ----------------
    #
    #  Zakładamy, że w pliku _load_model jest analogiczna funkcja,
    #  ale musimy upewnić się, że ładuje model ViT z 2 wyjściami:
    #
    #  def vit():
    #      model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    #      model.classifier = nn.Linear(model.classifier.in_features, 2)
    #      return model
    #
    model = _load_model(model_path=MODEL_PATH)
    model = model.to(device)

    # ---------------- 4) DataLoaders ----------------
    train_loader = _prepare_dataset_loader(IMAGES_LIST_TXT, batch_size=BATCH_SIZE)
    test_loader  = _prepare_dataset_loader(TEST_LIST_TXT,  batch_size=BATCH_SIZE)

    model.eval()
    acc_before, _ = evaluate_model(model, test_loader, suppres_printing=False)
    print(f"Accuracy before BPFA pruning: {acc_before:.3f}")

    # ---------------- 5) Rejestrujemy HOOK ----------------
    hook_handle = model.classifier.register_forward_hook(classifier_input_hook)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, races = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            feature_storage["preds_list"].extend(preds.tolist())
            feature_storage["races_list"].extend(races)
            feature_storage["labels_list"].extend(labels.numpy().tolist())

    # Teraz zebrane aktywacje:
    x_input_cat = torch.cat(feature_storage["x_input_list"], dim=0)  # (N, in_features)
    preds_np    = np.array(feature_storage["preds_list"])            # (N,)
    races_np    = np.array(feature_storage["races_list"])            # (N,)
    labels_np   = np.array(feature_storage["labels_list"])           # (N,)

    # Usuwamy hook – już niepotrzebny
    hook_handle.remove()

    # ---------------- 6) Obliczamy "bias" i wykonujemy pruning ----------------
    bias_array = compute_bias_per_input_dim(x_input_cat, races_np)   # shape: (in_features,)

    # Pruning w warstwie linear 'classifier'
    bpfa_pruning_linear_layer(model.classifier, bias_array, pruning_rate=PRUNING_RATE)

    # ---------------- 7) Ocena po modyfikacjach wag ----------------
    acc_after, _ = evaluate_model(model, test_loader, suppres_printing=False)
    print(f"Accuracy after BPFA pruning: {acc_after:.3f}")

    # ---------------- 8) Zapisujemy zmodyfikowany model -------------
    torch.save(model.state_dict(), "my_vit_after_bpfa_pruning.pth")
    print("Finished all steps.")
