UNLABELED_DIR = "/content/unlabeled_mri/"
TRAIN_DIR = "/content/train/"
VAL_DIR = "/content/val/"

PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 5

MAE_SAVE_PATH = "mae_encoder.pth"

# ---------------------------
# Dataset 1: binary brain tumor
# ---------------------------
DATASET1_NAME = "binary_brain_tumor"
DATASET1_TRAIN_PATH = "/content/training_images"
DATASET1_TEST_PATH = "/content/testing_images"
DATASET1_NUM_CLASSES = 2

# ---------------------------
# Dataset 2: 4-class brain tumor
# ---------------------------
DATASET2_NAME = "4class_brain_tumor"
DATASET2_KAGGLE_PATH = "sartajbhuvaji/brain-tumor-classification-mri"
DATASET2_NUM_CLASSES = 4

# ---------------------------
# Common training settings
# ---------------------------
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 224
