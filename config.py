UNLABELED_DIR = "/content/unlabeled_mri/"
TRAIN_DIR = "/content/train/"
VAL_DIR = "/content/val/"

PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 5

#----------------------------
# MAE Settings
#----------------------------
MAE_LEARNING_RATE = 1e-5
MAE_BATCH_SIZE = 32
MAE_EPOCHS = 10
MAE_IMG_SIZE = 224
MAE_PATCH_SIZE = 16
MAE_ENCODER_DIM = 768
MAE_ENCODER_DEPTH = 12
MAE_ENCODER_HEADS = 12
MAE_DECODER_DIM = 512
MAE_DECODER_DEPTH = 8
MAE_DECODER_HEADS = 16
MAE_MASK_RATIO = 0.125
MAE_ENCODER_SAVE_PATH = "./models/mae_encoder.pth"
MAE_FULL_SAVE_PATH = "./models/mae_full.pth"

#------------------------------
# Classifier Settings
#------------------------------ 
CLASSIFIER_LEARNING_RATE = 1e-5
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 30
CLASSIFIER_WEIGHT_DECAY = 0.01
CLASSIFIER_SAVE_PATH = "./models/classifier.pth"
AUGMENTATION = True

#----------------------------
# Selected Dataset
#----------------------------
DATASET = "dataset1"

# ---------------------------
# Dataset 1: binary brain tumor
# ---------------------------
DATASET1_NAME = "binary_brain_tumor"
DATASET1_PATH = "https://drive.google.com/uc?id=1RK-qLscrh8WirgwSxOcTbq5_3F3UcpTa"
DATASET1_NUM_CLASSES = 2

# ---------------------------
# Dataset 2: 4-class brain tumor
# ---------------------------
DATASET2_NAME = "brain-tumor-mri-dataset"
#DATASET2_KAGGLE_PATH = "sartajbhuvaji/brain-tumor-classification-mri"
DATASET2_KAGGLE_PATH = "masoudnickparvar/brain-tumor-mri-dataset"
DATASET2_NUM_CLASSES = 4
 
# ---------------------------
# Common training settings
# ---------------------------
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 224
