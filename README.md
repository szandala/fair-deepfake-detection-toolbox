# Fair DeepFake detection toolbox

# Fair Deepfake Image Detection Tools (AI Models)

A repository dedicated to developing and evaluating fair deepfake image detection models. This toolkit includes functions to measure model accuracy and fairness metrics across different demographic groups, and provides implementations for standard training, in-process and post-process fairness adjustments.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)

- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training Models](#training-models)
    - [Standard Training](#standard-training)
    - [In-Process Fair Training](#in-process-fair-training)
    - [Post-Process Fairness Training](#post-process-fairness-training)
  - [Evaluating Fairness Metrics](#evaluating-fairness-metrics)
- [Fairness Metrics Explained](#fairness-metrics-explained)
  - [Equality of Odds Parity](#equality-of-odds-parity)
  - [Predictive Value Parity](#predictive-value-parity)
- [License](#license)

---

## Features

- **Accuracy Measurement**: Evaluate the overall performance of deepfake detection models.
- **Fairness Metrics**: Functions to compute fairness metrics such as Equality of Odds Parity and Predictive Value Parity for the following demographic groups:
  - Asian
  - Black
  - Middle Eastern
  - Indian
  - Latino
  - White
- **Training Implementations**:
  - **Standard Training**: Baseline model training without fairness considerations.
  - **Pre-Process Fairness Training**: Apply fairness adjustments to the training dataset.
  - **In-Process Fair Training**: Incorporate fairness constraints directly into the training process.
  - **Post-Process Fairness Adjustment**: Apply fairness adjustments after model training.

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Torchvision
- Additional Python packages listed in `requirements.txt`


## Usage

### Data Preparation

I have prepared dataset from Kaggle: TODO put URL here


### Training Models

#### Standard Training

Run the standard training script to train a baseline model:

```bash
python train_model.py
```

#### In-Process Fair Training

Train a model with fairness constraints incorporated into the learning process:

```bash
python train_inprocess_fairness.py
```

<!-- #### Post-Process Fairness Training

Adjust a pre-trained model to improve fairness metrics:

```bash
python train_postprocess_fairness.py
``` -->

### Evaluating Fairness Metrics

After training, evaluate the model's performance and fairness:

```bash
python evaluate_fairness.py
```

This script will output confusion matrix and fairness metrics for each demographic group.

---

## Fairness Metrics Explained

### Equality of Odds Parity

Equality of Odds Parity assesses whether the model's true positive and false positive rates are equal across different demographic groups. A model satisfies Equality of Odds if these rates are the same for all groups.

### Predictive Value Parity

Predictive Value Parity checks if the positive predictive value (precision) and negative predictive value are equal across demographic groups. This ensures that the likelihood of a prediction being correct is the same for all groups.

---

## License

This project is licensed under the [MIT License](LICENSE).
