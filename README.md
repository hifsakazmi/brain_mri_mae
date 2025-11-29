# MAE Based Brain Tumor Classification

## Project Overview
This project focuses on brain tumor classification from MRI scans, a task made challenging by limited and imbalanced medical datasets. We re-implemented the SSCLNet contrastive learning framework and evaluated it on both 2-class and 4-class tumor datasets from Kaggle. To improve performance, we replaced the ResNet contrastive encoder with a Masked Autoencoder (MAE) backbone and an MLP classifier. The goal is to assess whether MAE provides stronger representations and boosts classification accuracy compared to contrastive learning.

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
3. Open `brain_mri_mae_notebook.ipynb` notebook in Colab.
4. Mount google drive and install dependencies.
   
### Pre-train MAE
```python
%cd /content/brain_mri_mae
!python run_pretrain.py
```
MAE will be trained for number of epochs defined in `config.py` and MAE Model and MAE Encoder with the lease validation loss will be saved in google drive. 

### Fine-tune Classification
```python
%cd /content/brain_mri_mae
!python run_finetune.py
```
Classifier will be finetuned for number of epochs defined in `config.py` and model with the best validation accuracy will be saved mounted google drive. 

### Test the Classifier
```python
%cd /content/brain_mri_mae
!python test_classifier.py
```
This will load the fine-tuned classifier saved from google drive and run it on test dataset. 

