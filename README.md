# Fair DeepFake detection toolbox

# Fair Deepfake Image Detection Tools (AI Models)

A repository dedicated to developing and evaluating fair deepfake image detection models. This toolkit includes functions to measure model accuracy and fairness metrics across different demographic groups, and provides implementations for standard training, in-process and post-process fairness adjustments.

## Table of Contents

- [Features](#features)
- [Results](#results)
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

## Results

| Metric                         |  Std. model | Pre-Processing | In-Processing | Post-Processing |
|--------------------------------|-------------|----------------|---------------|-----------------|
| Accuracy (asian)               | 0.68       | 0.53           | 0.62          |                 |
| Accuracy (black)               | 0.66       | 0.56           | 0.61          |                 |
| Accuracy (indian)              | 0.69       | 0.51           | 0.64          |                 |
| Accuracy (latino_hispanic)     | 0.79       | 0.58           | 0.70          |                 |
| Accuracy (middle_eastern)      | 0.67       | 0.52           | 0.61          |                 |
| Accuracy (white)               | 0.79       | 0.58           | 0.66          |                 |
| True Positive Parity (asian)   | 0.92       | 0.25           | 0.91          |                 |
| True Positive Parity (black)   | 0.87       | 0.28           | 0.87          |                 |
| True Positive Parity (indian)  | 0.92       | 0.29           | 0.93          |                 |
| True Positive Parity (latino_hispanic) | 0.87 | 0.21 | 0.86          |                 |
| True Positive Parity (middle_eastern)  | 0.84 | 0.24 | 0.85          |                 |
| True Positive Parity (white)   | 0.91       | 0.23           | 0.91          |                 |
| False Positive Parity (asian)  | 0.92       | 0.23           | 0.89          |                 |
| False Positive Parity (black)  | 0.87       | 0.26           | 0.87          |                 |
| False Positive Parity (indian) | 0.92       | 0.32           | 0.91          |                 |
| False Positive Parity (latino_hispanic) | 0.87 | 0.23 | 0.86         |                 |
| False Positive Parity (middle_eastern)  | 0.84 | 0.22 | 0.84         |                 |
| False Positive Parity (white)  | 0.91       | 0.24           | 0.88          |                 |
| Positive Predictive Value (asian)  | 0.46       | 0.48           | 0.56          |                 |
| Positive Predictive Value (black)  | 0.51       | 0.48           | 0.61          |                 |
| Positive Predictive Value (indian) | 0.50       | 0.64           | 0.60          |                 |
| Positive Predictive Value (latino_hispanic)| 0.69 | 0.65 | 0.79      |                 |
| Positive Predictive Value (middle_eastern) | 0.47 | 0.50 | 0.57       |                 |
| Positive Predictive Value (white)  | 0.63       | 0.64           | 0.74          |                 |
| Negative Predictive Value (asian)  | 0.90       | 0.65           | 0.90          |                 |
| Negative Predictive Value (black)  | 0.75       | 0.56           | 0.77          |                 |
| Negative Predictive Value (indian) | 0.86       | 0.70           | 0.90          |                 |
| Negative Predictive Value (latino_hispanic) | 0.67 | 0.60 | 0.68      |                 |
| Negative Predictive Value (middle_eastern)  | 0.78 | 0.62 | 0.79       |                 |
| Negative Predictive Value (white)  | 0.77       | 0.59           | 0.78          |                 |


| Model overall (min/max)   | Std. model | Pre-Processing | In-Processing | Post-Processing |
|---------------------------|------------|----------------|---------------|-----------------|
| Accuracy                  | 0.84       | 0.88           | 0.87          |                 |
| True Positive Parity      | 0.91       | 0.72           | 0.91          |                 |
| False Positive Parity     | 0.91       | 0.69           | 0.92          |                 |
| Positive Predictive Value | 0.67       | 0.74           | 0.71          |                 |
| Negative Predictive Value | 0.74       | 0.80           | 0.76          |                 |

### Conclusions
- These are dummy models, ligthly trained.
- Pre-processing gives the most **fair** result, but overall accuracy is creep- this is due to limited size of dataset.
- In-processing provides decent (not know whether significant already) fairness increase at the cost of accuracy.

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
