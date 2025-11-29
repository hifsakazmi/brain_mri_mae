# MAE Based Brain Tumor Classification

## Project Overview
Brain tumor ` from MRI images is a critical task for early diagnosis and treatment planning. However, medical imaging datasets are often small, imbalanced, and expensive to annotate. Traditional supervised deep learning models struggle in such settings because they rely heavily on large labeled datasets.

The base paper, SSCLNet: A Self-Supervised Contrastive Loss-Based Pre-Trained Network for Brain MRI Classification, proposes a contrastive learning framework using ResNet encoders to improve feature learning from unlabeled MRI data. This method enhances representation learning and improves downstream classification performance.

In this project, we re-implemented the SSCLNet framework and evaluated it on two datasets:
Brain Tumor 2-Class (tumor vs. no-tumor)
Brain Tumor 4-Class (glioma, meningioma, pituitary, no-tumor)

As an improvement, we replaced the ResNet-based contrastive encoder with a Masked Autoencoder (MAE) backbone and incorporated an MLP classifier head. Our objective was to evaluate whether MAE produces richer representations compared to contrastive learning encoders and can therefore improve tumor classification accuracy and robustness.

## How to run this code? 
1. Fork this repo.
2. Set following parameters in `config.py`
   * Select the dataset you want your model to train on (4-Class or 2-Class)
```python
# set DATASET = "dataset1" for 2-Class Dataset
DATASET = "dataset2" # This is 4-Class Dataset
```
  * Set Learning Rate for 4-Class Dataset
```python
MAE_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 4e-4
```
  * Set Learning Rate for 2-Class Dataset
```python
MAE_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-5
```
