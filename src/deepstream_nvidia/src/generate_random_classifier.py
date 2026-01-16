import torch
import torch.nn as nn
import torch.onnx

# Configuration
OUTPUT_ONNX_FILE = "random_classifier_91_classes.onnx"
NUM_CLASSES = 91
# Standard dummy input shape (Batch=1, RGB=3, H=224, W=224)
# You can adjust height/width here if your DeepStream config requires specific dims
INPUT_CHANNELS = 3
INPUT_HEIGHT = 224
INPUT_WIDTH = 224

class RandomClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RandomClassifier, self).__init__()
        # A simple feature extractor
        # Input: [B, 3, H, W]
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Downsample
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Forces output to [B, 32, 1, 1] regardless of input size
            nn.Flatten()                  # [B, 32]
        )
        # The classification head
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        # DeepStream classifiers usually expect [Batch, Num_Classes]
        x = self.classifier(x)
        return x

def export_model():
    # 1. Initialize model (weights are random by default)
    model = RandomClassifier(NUM_CLASSES)
    model.eval()

    # 2. Create dummy input for the tracer
    dummy_input = torch.randn(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

    # 3. Export to ONNX
    print(f"Exporting model to {OUTPUT_ONNX_FILE}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        OUTPUT_ONNX_FILE,
        export_params=True,
        opset_version=11,          # Opset 11 is generally very stable for DeepStream/TensorRT
        do_constant_folding=True,
        input_names=['input'],     # Standard input name
        output_names=['species_head'], # SPECIFIC REQUEST: Output layer name
        dynamic_axes={
            'input': {0: 'batch_size'},       # Variable batch size
            'species_head': {0: 'batch_size'} # Variable batch size
        }
    )
    
    print("Success! Model generated.")
    print(f"Input Name: input")
    print(f"Output Name: species_head")
    print(f"Output Shape: [batch_size, {NUM_CLASSES}]")

if __name__ == "__main__":
    export_model()