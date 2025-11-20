import os
import pandas as pd
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from skimage.feature import local_binary_pattern
from scipy.signal import convolve2d
from sklearn.model_selection import cross_val_score, StratifiedKFold


import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# selecao das amostras
@dataclass
class PKLotSample:
    path: str
    lot: str
    weather: str
    day: str
    label: str # ocupado ou vazio
    
    
def load_pklot_segmented_for_lot(
    base_dir: str,
    lot_name: str,
    valid_labels=("Occupied", "Empty"),
) -> List[PKLotSample]:
    """
    Estrutura esperada:
    base_dir/
        UFPR04/
            Sunny/
                2012-09-20/
                    Occupied/
                    Empty/
            Rainy/
            Cloudy/
        PUCPR/
            ...

    Retorna uma lista de PKLotSample com path, lot, weather, day e label.
    """
    lot_dir = os.path.join(base_dir, lot_name)

    samples: List[PKLotSample] = []

    # subpasta = condição climática (Cloudy)
    for weather in os.listdir(lot_dir):
        weather_dir = os.path.join(lot_dir, weather)
        if not os.path.isdir(weather_dir):
            continue

        # subpasta = dia (2012-09-20)
        for day in os.listdir(weather_dir):
            day_dir = os.path.join(weather_dir, day)
            if not os.path.isdir(day_dir):
                continue

            for label in os.listdir(day_dir):
                if label not in valid_labels:
                    continue

                label_dir = os.path.join(day_dir, label)
                if not os.path.isdir(label_dir):
                    continue

                for fname in os.listdir(label_dir):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        continue

                    full_path = os.path.join(label_dir, fname)
                    samples.append(
                        PKLotSample(
                            path=full_path,
                            lot=lot_name,
                            weather=weather,
                            day=day,
                            label=label,
                        )
                    )

    return samples


def _sample_by_class(
    samples: List[PKLotSample],
    label: str,
    n: int,
    rng: random.Random,
) -> List[PKLotSample]:
    subset = [s for s in samples if s.label == label]
    if len(subset) < n:
        raise ValueError(
            f"Classe insuficiente: '{label}'. "
            f"Requisitado: {n}, disponível: {len(subset)}"
        )
    return rng.sample(subset, n)


def make_train_test_split_for_lot(
    base_dir: str,
    lot_name: str,
    n_train_per_class: int = 3500, # valor solicitado professor
    n_test_per_class: int = 1000, # valor solicitado professor
    n_train_days: int = 10, # valor solicitado professor
    seed: int = 42,
) -> Tuple[List[PKLotSample], List[PKLotSample], List[str], List[str]]:
    rng = random.Random(seed)

    samples = load_pklot_segmented_for_lot(base_dir, lot_name)

    # ordena dias unicos
    unique_days = sorted({s.day for s in samples})
    if len(unique_days) <= n_train_days:
        raise ValueError(
            f"Estacionamento {lot_name} - {len(unique_days)} dias, "
        )

    train_days = unique_days[:n_train_days]
    test_days = unique_days[n_train_days:]

    # get treino e teste possiveis
    train_candidates = [s for s in samples if s.day in train_days]
    test_candidates = [s for s in samples if s.day in test_days]

    # separa por classe
    train_occ = _sample_by_class(train_candidates, "Occupied", n_train_per_class, rng)
    train_emp = _sample_by_class(train_candidates, "Empty", n_train_per_class, rng)

    test_occ = _sample_by_class(test_candidates, "Occupied", n_test_per_class, rng)
    test_emp = _sample_by_class(test_candidates, "Empty", n_test_per_class, rng)

    train_samples = train_occ + train_emp
    test_samples = test_occ + test_emp

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples, train_days, test_days


def print_label_distribution(name, samples):
    counter = Counter(s.label for s in samples)
    total = len(samples)
    print(f"\n{name}:")
    print(f"  Total: {total}")
    for label in sorted(counter.keys()):
        count = counter[label]
        print(f"  {label:8s}: {count:5d}  ({count/total:.2%})")
        
        

# obtencao de caracteristicas
class PKLotTorchDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.label_map = {"Empty": 0, "Occupied": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.label_map[s.label]
        return img, label

# TODO: definir transformacoes e verificar numeros corretos
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet
        std=[0.229, 0.224, 0.225]
    ),
])


def build_vgg16_feature_extractor():
    vgg = models.vgg16(pretrained=True)
    # remove ultima camada
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    # saida: dimensao 4096
    return vgg


def build_resnet50_feature_extractor():
    resnet = models.resnet50(pretrained=True)
    # remove a ultima camada
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules)

    class ResNet50Features(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            return x
    # saida: dimensao 2048
    return ResNet50Features(backbone)


def build_mobilenet_v2_feature_extractor():
    mobilenet = models.mobilenet_v2(pretrained=True)
    # remove a ultima camada com substituicao por identidade
    mobilenet.classifier = nn.Identity()
    # saida: dimensao 1280
    return mobilenet


# helper para construir todos os extratores de uma vez
def build_all_feature_extractors(device):
    models_dict = {
        "vgg16": build_vgg16_feature_extractor(),
        "resnet50": build_resnet50_feature_extractor(),
        "mobilenet_v2": build_mobilenet_v2_feature_extractor(),
    }
    for m in models_dict.values():
        m.to(device)
        m.eval()
    return models_dict


# extrair features usando um modelo
@torch.no_grad()
def extract_features_single_model(model, dataloader, device):
    """
    Extrai features e rótulos usando model para todas as amostras
    Retorna:
        X: np.array shape (n_samples, d_feature)
        y: np.array shape (n_samples,)
    """
    all_feats = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.numpy()

        feats = model(imgs)
        feats = feats.cpu().numpy()

        all_feats.append(feats)
        all_labels.append(labels)

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


# extrair features para um split
def compute_cnn_features_for_split(samples, batch_size=32, num_workers=4):
    """
    Retorna:
        features_dict: {
            "vgg16": np.array (n_samples, 4096),
            "resnet50": np.array (n_samples, 2048),
            "mobilenet_v2": np.array (n_samples, 1280),
        }
        y: np.array (n_samples,)
    """
    dataset = PKLotTorchDataset(samples, transform=img_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,      # mantem a ordem das amostras
        num_workers=num_workers,
        pin_memory=False,
    )

    models_dict = build_all_feature_extractors(device)

    features_dict = {}
    y_ref = None

    for name, model in models_dict.items():
        print(f"Extraindo features com {name}...")
        X, y = extract_features_single_model(model, dataloader, device)

        features_dict[name] = X
        if y_ref is None:
            y_ref = y
        else:
            # os rotulos devem ser os mesmos
            assert np.array_equal(y_ref, y), "Inconsistência de rótulos entre extrações!"

    return features_dict, y_ref


# codigos para fusoes
# contatenacao simples
def fuse_concat(features_dict):
    Xs = [feat for feat in features_dict.values()]
    return np.concatenate(Xs, axis=1)

# truncagem para menor dimensao
def _truncate_to_min_dim(features_dict):
    min_dim = min(feat.shape[1] for feat in features_dict.values())
    truncated = [feat[:, :min_dim] for feat in features_dict.values()]
    return np.stack(truncated, axis=0)  # (n_models, n_samples, min_dim)

# media
def fuse_mean(features_dict):
    stack = _truncate_to_min_dim(features_dict)    
    X_mean = stack.mean(axis=0)                  
    return X_mean

# soma
def fuse_sum(features_dict):
    stack = _truncate_to_min_dim(features_dict)    
    X_sum = stack.sum(axis=0)                    
    return X_sum

# fusao ponderada
def fuse_weighted(features_dict, weights):
    """
    precisa definir os pesos em dict:
        {"vgg16": 0.5, "resnet50": 0.3, "mobilenet_v2": 0.2}
    """
    keys = list(features_dict.keys())

    # pesos precisam somar 1
    w = np.array([weights[k] for k in keys], dtype=np.float32)
    w = w / w.sum()

    # truncagem
    min_dim = min(features_dict[k].shape[1] for k in keys)
    truncated = [features_dict[k][:, :min_dim] for k in keys]
    stack = np.stack(truncated, axis=0)

    w = w.reshape(-1, 1, 1)
    X_weighted = (stack * w).sum(axis=0)  # (n, d)

    return X_weighted


# treinamento dos modelos
def train_models(
    datasets_fusions: Dict[str, Dict[str, tuple]],
    random_state: int = 42,
):
    """
    Parâmetros:
      datasets_fusions:
        {
          "UFPR04": {
              "concat":   (X_train_concat_ufpr04,   X_test_concat_ufpr04,   ufpr04_train_y, ufpr04_test_y),
              "mean":     (X_train_mean_ufpr04,     X_test_mean_ufpr04,     ufpr04_train_y, ufpr04_test_y),
              "sum":      (X_train_sum_ufpr04,      X_test_sum_ufpr04,      ufpr04_train_y, ufpr04_test_y),
              "weighted": (X_train_weighted_ufpr04, X_test_weighted_ufpr04, ufpr04_train_y, ufpr04_test_y),
          },
          "PUC": {
              "concat":   (X_train_concat_pucpr,   X_test_concat_pucpr,   pucpr_train_y, pucpr_test_y),
              "mean":     (X_train_mean_pucpr,     X_test_mean_pucpr,     pucpr_train_y, pucpr_test_y),
              "sum":      (X_train_sum_pucpr,      X_test_sum_pucpr,      pucpr_train_y, pucpr_test_y),
              "weighted": (X_train_weighted_pucpr, X_test_weighted_pucpr, pucpr_train_y, pucpr_test_y),
          },
        }

    Retorna:
      models_dict:
        chave: (dataset_name, algoritmo, tipo_fusao)
        valor: modelo treinado (objeto sklearn)
    """

    models_dict = {}

    for dataset_name, fusions_dict in datasets_fusions.items():
        print(f"\nDataset: {dataset_name}")

        for fusion_name, (X_train, X_test, y_train, y_test) in fusions_dict.items():
            print(f"Treinando modelos para fusão: {fusion_name} | shape: {X_train.shape}")

            # Modelo SVM
            svm = SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=False, 
            )
            svm.fit(X_train, y_train)
            models_dict[(dataset_name, "SVM", fusion_name)] = svm
            print(f"SVM treinado ({dataset_name}, {fusion_name})")

            # Modelo MLP
            mlp = MLPClassifier(
                hidden_layer_sizes=(512,),
                activation="relu",
                solver="adam",
                max_iter=50,
                random_state=random_state,
            )
            mlp.fit(X_train, y_train)
            models_dict[(dataset_name, "MLP", fusion_name)] = mlp
            print(f"MLP treinado ({dataset_name}, {fusion_name})")

    print("\nTreinamento concluído.")
    print(f"Total de modelos treinados: {len(models_dict)}")
    return models_dict


# avaliacao dos modelos
def evaluate_models(
    models_dict,
    datasets_fusions,
    average_key: str = "weighted avg",
):
    """
    Avalia todos os modelos em:
        intra-dataset 
        cross-dataset

    average_key:
      qual linha do classification_report usar para o F1:
      weighted avg ou macro avg

    Retorna:
      df_results: DataFrame com F1 para cada combinação.
    """

    results = []

    dataset_names = list(datasets_fusions.keys())
    algos = ["SVM", "MLP"]

    for train_dataset in dataset_names:
        for fusion_name in datasets_fusions[train_dataset].keys():
            for algo in algos:
                model_key = (train_dataset, algo, fusion_name)
                if model_key not in models_dict:
                    continue

                model = models_dict[model_key]

                for eval_dataset in dataset_names:
                    # pega o X_test e y_test do dataset de avaliação
                    _, X_test, _, y_test = datasets_fusions[eval_dataset][fusion_name]

                    y_pred = model.predict(X_test)

                    report = classification_report(
                        y_test,
                        y_pred,
                        target_names=["Empty", "Occupied"],
                        output_dict=True,
                        zero_division=0,
                    )

                    f1 = report[average_key]["f1-score"]

                    results.append({
                        "train_dataset": train_dataset,
                        "eval_dataset": eval_dataset,
                        "algo": algo,
                        "fusion": fusion_name,
                        "f1_" + average_key.replace(" ", "_"): f1,
                        "f1_macro": report["macro avg"]["f1-score"],
                        "f1_weighted": report["weighted avg"]["f1-score"],
                    })

    df_results = pd.DataFrame(results)
    return df_results


# matriz de confusao
def plot_confusion_matrix_for_model(
    models_dict,
    datasets_fusions,
    train_dataset: str,
    eval_dataset: str,
    fusion_name: str,
    algo: str,
    normalize: str | None = None,  # None, "true", "pred", "all"
):
    model_key = (train_dataset, algo, fusion_name)

    model = models_dict[model_key]

    _, X_test, _, y_test = datasets_fusions[eval_dataset][fusion_name]

    y_pred = model.predict(X_test)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Empty", "Occupied"],
        normalize=normalize,
        cmap="Blues",
    )
    title = f"{algo} - {fusion_name}\ntrain={train_dataset}, eval={eval_dataset}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


# adicoes alem das funcoes pedidas

# implementando LBP
def compute_lbp_hist(gray_img: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """
    gray_img: imagem em escala de cinza (np.uint8) 2D
    Retorna: histograma LBP normalizado (shape: (59,))
    """
    lbp = local_binary_pattern(gray_img, P, R, method="uniform")

    n_bins = P * (P - 1) + 3  # = 59 para P=8
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(n_bins + 1),
        range=(0, n_bins),
        density=True,  # normalizado
    )
    return hist.astype(np.float32)


# nova funcao para features
def compute_texture_features_for_split(
    samples,
    resize_to: Tuple[int, int] | None = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Versão alternativa a compute_cnn_features_for_split,
    """
    label_map = {"Empty": 0, "Occupied": 1}

    X_lbp_list = []
    y_list = []

    for s in samples:
        img = Image.open(s.path).convert("L")  # escala de cinza

        if resize_to is not None:
            img = img.resize(resize_to, Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)

        lbp_hist = compute_lbp_hist(img_np, P=8, R=1)

        X_lbp_list.append(lbp_hist)
        y_list.append(label_map[s.label])

    X_lbp = np.vstack(X_lbp_list)
    y = np.array(y_list, dtype=np.int64)

    features_dict = {
        "lbp": X_lbp,
    }
    return features_dict, y


# validacao cruzada dos modelos
def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    algo_name: str = "SVM",
    random_state: int = 42,
    cv_splits: int = 5,
):
    """
    Faz validação cruzada estratificada para um único algoritmo (SVM ou MLP)
    em um conjunto de treino (X, y).

    Retorna:
      (mean_f1, std_f1)
    """

    if algo_name == "SVM":
        model = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=False,
        )
    elif algo_name == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=(512,),
            activation="relu",
            solver="adam",
            max_iter=50,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Algo desconhecido: {algo_name}")

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=2,
    )

    return scores.mean(), scores.std()


def cross_validate_all(
    datasets_fusions,
    random_state: int = 42,
    cv_splits: int = 5,
):
    """
    Roda validação cruzada (apenas no treino) para:
      - cada dataset (UFPR04, PUC, ...)
      - cada fusão (concat, mean, sum, weighted)
      - cada algoritmo (SVM, MLP)

    Usando a função cross_validate_model.

    Retorna:
      df_cv: DataFrame com F1 médio e desvio padrão.
    """
    results = []
    algos = ["SVM", "MLP"]

    for dataset_name, fusions_dict in datasets_fusions.items():
        for fusion_name, (X_train, X_test, y_train, y_test) in fusions_dict.items():
            for algo in algos:
                mean_f1, std_f1 = cross_validate_model(
                    X_train,
                    y_train,
                    algo_name=algo,
                    random_state=random_state,
                    cv_splits=cv_splits,
                )

                results.append({
                    "dataset": dataset_name,
                    "fusion": fusion_name,
                    "algo": algo,
                    "cv_splits": cv_splits,
                    "f1_cv_mean": mean_f1,
                    "f1_cv_std": std_f1,
                })
                print(f"[CV] {dataset_name} | {fusion_name} | {algo} -> F1={mean_f1:.4f} ± {std_f1:.4f}")

    df_cv = pd.DataFrame(results)
    return df_cv
