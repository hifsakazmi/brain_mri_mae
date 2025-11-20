import config
from training.finetune_classifier import finetune_classifier

if __name__ == "__main__":
    # Fine-tune on dataset2 with pre-trained MAE
    finetune_classifier(
        config, 
        dataset_name=config.DATASET, 
        use_drive_checkpoint=True
    )