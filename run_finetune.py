import config
from training.finetune_classifier import finetune_classifier

if __name__ == "__main__":
    # Fine-tune on dataset2 with pre-trained MAE
    finetune_classifier(
        config, 
        dataset_name="dataset2", 
        pretrained_path=config.MAE_ENCODER_SAVE_PATH
    )