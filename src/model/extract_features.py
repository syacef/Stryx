
from model.dinov2_extractor import DINOv2Extractor

if __name__ == "__main__":
    extractor = DINOv2Extractor(model_type='dinov2_vits14')
    extractor.extract_features(
        input_dir="./data",
        output_dir="./data/embeddings/dinov2_vits14", 
        batch_size=64,
        num_workers=4
    )
