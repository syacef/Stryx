import random
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image


class BboxAwareCrop:
    """
    Crop that focuses on bounding boxes 70% of the time, random 30% of the time.
    
    Args:
        output_size: Target crop size (int or tuple)
        bbox_prob: Probability of doing bbox-aware crop (default: 0.7)
        margin_range: Range of margin around bbox as multiplier (default: (1.2, 2.0))
    """
    
    def __init__(self, output_size, bbox_prob=0.7, margin_range=(1.2, 2.0)):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.bbox_prob = bbox_prob
        self.margin_range = margin_range
    
    def __call__(self, img, bbox=None):
        """
        Args:
            img: PIL Image
            bbox: Dict with keys 'x', 'y', 'width', 'height' (in pixel coordinates)
                  If None or random probability, does random crop
        
        Returns:
            Cropped PIL Image
        """
        width, height = img.size
        target_h, target_w = self.output_size
        
        # 30% random crop or if no bbox provided
        if bbox is None or random.random() > self.bbox_prob:
            return self._random_crop(img, target_h, target_w)
        
        # 70% bbox-aware crop
        return self._bbox_crop(img, bbox, target_h, target_w)
    
    def _random_crop(self, img, target_h, target_w):
        """Standard random crop"""
        width, height = img.size
        
        if width < target_w or height < target_h:
            # If image smaller than target, resize first
            img = F.resize(img, max(target_h, target_w))
            width, height = img.size
        
        top = random.randint(0, max(0, height - target_h))
        left = random.randint(0, max(0, width - target_w))
        
        return F.crop(img, top, left, target_h, target_w)
    
    def _bbox_crop(self, img, bbox, target_h, target_w):
        """Bbox-aware crop with random margin"""
        width, height = img.size
        
        # Extract bbox coordinates
        bbox_x = bbox['x']
        bbox_y = bbox['y']
        bbox_w = bbox['width']
        bbox_h = bbox['height']
        
        # Calculate bbox center
        center_x = bbox_x + bbox_w / 2
        center_y = bbox_y + bbox_h / 2
        
        # Determine crop size with random margin
        margin = random.uniform(*self.margin_range)
        crop_size = max(bbox_w, bbox_h) * margin
        
        # Ensure crop is at least target size
        crop_size = max(crop_size, max(target_h, target_w))
        
        # Ensure crop doesn't exceed image bounds
        crop_size = min(crop_size, min(width, height))
        
        # Calculate crop boundaries centered on bbox
        left = max(0, int(center_x - crop_size / 2))
        top = max(0, int(center_y - crop_size / 2))
        
        # Adjust if crop goes beyond image
        if left + crop_size > width:
            left = max(0, int(width - crop_size))
        if top + crop_size > height:
            top = max(0, int(height - crop_size))
        
        # Crop to square
        crop_h = crop_w = int(crop_size)
        
        # If crop size is smaller than target, resize the whole image first
        if crop_size < max(target_h, target_w):
            scale = max(target_h, target_w) / crop_size
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = F.resize(img, (new_height, new_width))
            
            # Recalculate crop position after resize
            left = int(left * scale)
            top = int(top * scale)
            crop_h = int(crop_h * scale)
            crop_w = int(crop_w * scale)
        
        # Perform crop
        cropped = F.crop(img, top, left, crop_h, crop_w)
        
        # Resize to target size
        cropped = F.resize(cropped, self.output_size)
        
        return cropped
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(output_size={self.output_size}, "
                f"bbox_prob={self.bbox_prob}, margin_range={self.margin_range})")


class BboxAwareTransform:
    """
    Wrapper that applies bbox-aware crop followed by standard transforms.
    Works as a callable transform in the pipeline.
    """
    
    def __init__(self, bbox_crop, additional_transforms=None):
        """
        Args:
            bbox_crop: BboxAwareCrop instance
            additional_transforms: torchvision.transforms.Compose or list of transforms
        """
        self.bbox_crop = bbox_crop
        self.additional_transforms = additional_transforms
    
    def __call__(self, img, bbox=None):
        """
        Args:
            img: PIL Image
            bbox: Optional bbox dict
        
        Returns:
            Transformed tensor
        """
        # Apply bbox-aware crop
        img = self.bbox_crop(img, bbox)
        
        # Apply additional transforms
        if self.additional_transforms:
            img = self.additional_transforms(img)
        
        return img
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"  bbox_crop={self.bbox_crop},\n"
                f"  additional_transforms={self.additional_transforms}\n"
                f")")
