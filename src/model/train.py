import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from model.ssl.pipeline import SafariPipeline

if __name__ == "__main__":
    RAW_DATA_PATH = "./data"
    DINO_BACKBONE = "dinov2_vits14"

    pipeline = SafariPipeline(model_type=DINO_BACKBONE)
    pipeline.extract_features(RAW_DATA_PATH)
    pipeline.train_ssl(epochs=30)
