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

| Metric                     | Std. model | In-Processing | Pre-Processing - undersampling|
|----------------------------|------------|-------------|--------|
| **Overall Accuracy**       | 0.8866       | 0.8518    | 0.7561 |
| **True Positive Parity**   | 0.84         | 0.88      | 0.89   |
| **False Positive Parity**  | 0.57         | 0.72      | 0.76   |
| **Positive Predictive Value** | 0.84      | 0.81      | 0.79   |
| **Negative Predictive Value** | 0.91      | 0.92      | 0.83   |


### Conclusions


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

### Dataset Source

I have prepared dataset from Kaggle: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

```tex
@Inproceedings{ltnghia-ICCV2021,
  Title          = {OpenForensics: Large-Scale Challenging Dataset For
Multi-Face Forgery Detection And Segmentation In-The-Wild},
  Author         = {Trung-Nghia Le and Huy H. Nguyen and Junichi Yamagishi
and Isao Echizen},
  BookTitle      = {International Conference on Computer Vision},
  Year           = {2021},
}
```

Zip can be downloaded from m [GDrive](https://drive.google.com/file/d/1nduIze2HqEDmoT8FbByIDGsqW-9uHN-J/view?usp=sharing).

Gender and race attribution was done using `DeepFace` framework + manual corrections.

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
| Metric                    | Equivalent Term         | Focus               | Description                                                       | Formula            |
|---------------------------|-------------------------|---------------------|------------------------------------------------------------------|---------------------|
| Accuracy                  | -                       | Overall correct classifications | -                                                                | $\frac{TP + TN}{TP + TN + FP + FN}$|
| True Positive Parity      | Recall / Sensitivity    | Actual Positives    | High TPP indicates good recall and low false negatives.          | $\frac{TP}{TP + FN}$              |
| False Positive Parity     | False Positive Rate     | Actual Negatives    | Complementary to Specificity; balancing FPP prevents bias in negative misclassifications. | $\frac{FP}{FP + TN}$|
| Positive Predictive Value | Precision               | Predicted Positives | Balances with Recall; indicates low FP rate among positives.     | $\frac{TP}{TP + FP}$              |
| Negative Predictive Value | -                       | Predicted Negatives | Complements PPV; indicates low FN rate among negatives.          | $\frac{TN}{TN + FN}$              |
| Specificity               | True Negative Rate      | Actual Negatives    | Complements FPR; indicates accuracy in predicting negatives.     | $\frac{TN}{TN + FP}$               |


### Equality of Odds Parity

Equality of Odds Parity assesses whether the model's true positive and false positive rates are equal across different demographic groups. A model satisfies Equality of Odds if these rates are the same for all groups.\
Metrics:
- True Positive Parity
- False Positive Parity

### Predictive Value Parity

Predictive Value Parity checks if the positive predictive value (precision) and negative predictive value are equal across demographic groups. This ensures that the likelihood of a prediction being correct is the same for all groups.\
Metrics:
- Positive Predictive Value
- Negative Predictive Value

---

## License

This project is licensed under the [MIT License](LICENSE).
