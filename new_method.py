import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluate_fairness import evaluate_model, _prepare_dataset_loader, _load_model
from common import FairDataset

import numpy as np

# --------------------------------------------------------------
# 1. HOOK, by przechwycić wejście do 'model.classifier'
# --------------------------------------------------------------
feature_storage = {
    "x_input_list": [],  # wektory wejściowe do classifiera
    "labels_list": [],
    "races_list": [],
    "preds_list": []
}

def classifier_input_hook(module, input_, output):
    """
    Funkcja hook wywoływana ZA KAŻDYM RAZEM, gdy przechodzi batch przez
    `model.classifier(...)`.
    - 'module' to warstwa nn.Linear
    - 'input_' to krotka zawierająca (Tensor o shape [batch_size, in_features],)
    - 'output' to wynik forwarda (logits), ale do niczego tutaj nieużywany.

    Zapisujemy TYLKO input do globalnej listy, żeby potem obliczyć wariancję.
    UWAGA: Hook jest wywoływany w trakcie forwardu modelu.
    """
    x_input = input_[0].detach().cpu()  # shape: (batch_size, in_features)
    feature_storage["x_input_list"].append(x_input)


# --------------------------------------------------------------
# 2. Funkcja do liczenia odchylenia standardowego w podziale
#    na rasy i KLASY (0/1).
#    Zakładamy, że:
#      - preds to np.array kształtu (N,)  -> 0 lub 1
#      - races to np.array kształtu (N,)  -> np. "White", "Black", ...
#      - X_input to np.array kształtu (N, in_features)
#
#    Dla klasy 0 i 1 liczymy std wektorów X_input w podziale na rasy,
#    a następnie uśredniamy w poprzek ras.
# --------------------------------------------------------------
def compute_std_across_races(preds, races, X_input):
    """
    Zwraca:
      std_class0, std_class1 -> tensory o shape (in_features,),
      gdzie każda składowa to ŚREDNIE odchylenie standardowe w danym wymiarze
      w poprzek wszystkich ras, dla klasy 0 i klasy 1.
    """
    unique_races = np.unique(races)
    in_features = X_input.shape[1]

    # Podział na sub-zbiory wg klasy 0 i 1
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

    # Liczymy odchylenie std w każdym wymiarze osobno dla każdej rasy,
    # potem uśredniamy w poprzek ras.
    std_accum_0 = []
    for r in unique_races:
        idx_r = (R_class0 == r)
        if np.sum(idx_r) > 1:
            std_r = np.std(X_class0[idx_r], axis=0)  # shape: (in_features,)
            std_accum_0.append(std_r)
        else:
            # jeżeli brak próbek -> wektor zer
            std_accum_0.append(np.zeros(in_features, dtype=np.float32))

    std_class0 = np.mean(std_accum_0, axis=0) if len(std_accum_0) > 0 else np.zeros(in_features)

    std_accum_1 = []
    for r in unique_races:
        idx_r = (R_class1 == r)
        if np.sum(idx_r) > 1:
            std_r = np.std(X_class1[idx_r], axis=0)
            std_accum_1.append(std_r)
        else:
            std_accum_1.append(np.zeros(in_features, dtype=np.float32))

    std_class1 = np.mean(std_accum_1, axis=0) if len(std_accum_1) > 0 else np.zeros(in_features)

    return torch.from_numpy(std_class0), torch.from_numpy(std_class1)


# --------------------------------------------------------------
# 3. Funkcja do skalowania wag -> DODAJEMY odchylenie std
# --------------------------------------------------------------
def scale_last_layer_weights(model, std_class0, std_class1, alpha=0.5):
    """
    Modyfikuje wagi w model.classifier (nn.Linear z 2 wyjściami)
    w taki sposób, że:
      W[0, n] (waga neuronu odpow. klasie 0, wymiar n)
        <- W[0, n] + alpha * std_class0[n]
      W[1, n] (waga neuronu odpow. klasie 1, wymiar n)
        <- W[1, n] + alpha * std_class1[n]

    UWAGA: w praktyce dodawanie std do wag może być
    mało 'intuicyjne' – ale tu jest placeholder, jak chciałeś.
    """
    if not isinstance(model.classifier, nn.Linear):
        raise TypeError("Ostatnia warstwa modelu nie jest nn.Linear!")

    W = model.classifier.weight.data.clone()  # shape: (2, in_features)
    b = model.classifier.bias.data.clone()    # shape: (2,)

    in_features = W.shape[1]

    # Upewniamy się, że std_class0 i std_class1 mają wymiar (in_features,)
    if std_class0.shape[0] != in_features or std_class1.shape[0] != in_features:
        raise ValueError("std_class0/std_class1 nie pasują do in_features warstwy classifier.")

    # Dodajemy alpha * std_classX do wagi W[X, n]
    for n in range(in_features):
        # klasa 0
        W[0, n] += alpha * std_class0[n]
        # klasa 1
        W[1, n] += alpha * std_class1[n]

    model.classifier.weight.data.copy_(W)
    model.classifier.bias.data.copy_(b)

    print(">> Last layer weights have been modified by adding alpha * std.")


# --------------------------------------------------------------
# 4. Przykładowy SKRYPT główny (z treningiem i użyciem w/w funkcji)
# --------------------------------------------------------------
if __name__ == "__main__":


    # 1) Parametry
    MODEL_PATH = "model_full_train_e12_acc0.887.pth"
    N_EPOCHS = 2
    BATCH_SIZE = 32
    IMAGES_LIST_TXT = "work_on_train.txt"
    TEST_LIST_TXT = "work_on_test.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Tworzymy model.
    model = _load_model(model_path=MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 3) Rejestrujemy HOOK na warstwę classifier (Linear)
    hook_handle = model.classifier.register_forward_hook(classifier_input_hook)

    # 4) DataLoaders
    train_loader = _prepare_dataset_loader(IMAGES_LIST_TXT, batch_size=BATCH_SIZE)
    test_loader = _prepare_dataset_loader(TEST_LIST_TXT, batch_size=BATCH_SIZE)

    # not needed as I have trained model
    # # 5) Optymalizacja (prosty przykład)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # # 6) Trening
    # for epoch in range(N_EPOCHS):
    #     running_loss = 0.0
    #     model.train()
    #     for i, data in enumerate(train_loader):
    #         inputs, labels, _races = data
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)  # -> ViTForImageClassificationOutput
    #         # outputs.logits ma shape: (batch_size, 2)

    #         loss = criterion(outputs.logits, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         if i % 20 == 0:
    #             print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/100:.3f}")
    #             running_loss = 0.0

    #     scheduler.step()
    #     # Ewaluacja prosta
    #     acc, _ = evaluate_model(model, test_loader, suppres_printing=True)
    #     print(f"Epoch {epoch+1} - Test Accuracy: {acc:.3f}")

    # (opcjonalnie) Zapisz stan wytrenowanego modelu
    # torch.save(model.state_dict(), "my_vit_trained.pth")

    # 7) Zbieranie WARTOŚCI WEJŚĆ do classifiera – w trakcie jednego przebiegu
    #    *inference* na zbiorze testowym
    #    HOOK juz jest zarejestrowany, więc w pętli forward
    #    hook dołoży embeddings do feature_storage["x_input_list"].
    #    W tym samym czasie zapiszemy preds, races, itp. do list.
    # feature_storage["x_input_list"] = []  # czyścimy, bo były z treningu
    # feature_storage["preds_list"] = []
    # feature_storage["races_list"] = []
    # feature_storage["labels_list"] = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, races = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            # preds -> argmax(logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            feature_storage["preds_list"].extend(preds.tolist())
            feature_storage["races_list"].extend(races)
            feature_storage["labels_list"].extend(labels.numpy().tolist())

    # Teraz x_input_list to LISTA tensora per batch
    x_input_cat = torch.cat(feature_storage["x_input_list"], dim=0)  # (N, in_features)
    preds_np = np.array(feature_storage["preds_list"])               # (N,)
    races_np = np.array(feature_storage["races_list"])               # (N,)
    labels_np = np.array(feature_storage["labels_list"])             # (N,)

    # 8) Liczymy odchylenie standardowe w podziale na rasy i klasy 0/1
    std_class0, std_class1 = compute_std_across_races(preds_np, races_np, x_input_cat.numpy())
    print("std_class0 shape:", std_class0.shape)
    print("std_class1 shape:", std_class1.shape)

    # 9) Skalujemy wagi ostatniej warstwy -> dodajemy alpha * std
    scale_last_layer_weights(model, std_class0, std_class1, alpha=0.2)
    # (opcjonalnie) odpinamy hook, jeśli już nie jest potrzebny
    hook_handle.remove()

    # 10) Po modyfikacji wag -> ewentualnie znów oceniamy model
    acc_after, _ = evaluate_model(model, test_loader, suppres_printing=True)
    print(f"Accuracy after weight scaling: {acc_after:.3f}")

    # Na koniec można zapisać zmodyfikowany model
    torch.save(model.state_dict(), "my_vit_after_scaling.pth")


    print("Finished all steps.")
