from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class CaltechCameraTrapsDataset(Dataset):
    def __init__(self, image_dir, processor, augment=True):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.rglob("*.jpg")) + list(self.image_dir.rglob("*.png"))
        self.processor = processor
        self.augment = augment
        
        # Strong augmentations for SSL
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.augment:
            # Create two augmented views of the same image (for contrastive learning)
            view1 = self.transform(image)
            view2 = self.transform(image)
            
            # Process with DINOv2 processor
            inputs1 = self.processor(images=view1, return_tensors="pt")
            inputs2 = self.processor(images=view2, return_tensors="pt")
            
            return {
                'pixel_values_1': inputs1['pixel_values'].squeeze(0),
                'pixel_values_2': inputs2['pixel_values'].squeeze(0)
            }
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            return {'pixel_values': inputs['pixel_values'].squeeze(0)}
