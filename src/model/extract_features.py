
from model.dinov2_extractor import DINOv2Extractor

if __name__ == "__main__":
    DATASET_PATH = "./data"
    EMBEDDING_OUT = "./data/embeddings"

    extractor = DINOv2Extractor(model_type='dinov2_vits14')
    extractor.extract_features(
        input_dir=DATASET_PATH, 
        output_dir=EMBEDDING_OUT, 
        batch_size=64,
        num_workers=4
    )
