import os
import json
from torch.utils.data import Dataset
from PIL import Image
# --- CCT Dataset Parser ---
class CCTDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None, image_ids=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            json_file (string): Path to the CCT json file.
            transform (callable, optional): Transform to be applied on a sample.
            image_ids (list, optional): List of specific image IDs to include (for train/val split).
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load JSON
        print(f"Loading annotations from {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # 1. Map Image ID -> File Name
        self.images = {img['id']: img['file_name'] for img in data['images']}
        
        # 2. Map Category ID -> Category Name (for debug) & Index
        # CCT Category IDs might be [1, 5, 10], we need [0, 1, 2] for PyTorch
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
        sorted_cat_ids = sorted(self.cat_id_to_name.keys())
        self.cct_id_to_model_idx = {cct_id: idx for idx, cct_id in enumerate(sorted_cat_ids)}
        self.classes = [self.cat_id_to_name[c_id] for c_id in sorted_cat_ids]
        
        # 3. Map Image ID -> Category ID
        # Note: If an image has multiple boxes, this script takes the FIRST one.
        # Use this loop to handle empty images (if category_id 0 is 'empty' in your JSON)
        self.img_id_to_cat_id = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            # Only store if we haven't seen this image yet (single label assumption)
            if img_id not in self.img_id_to_cat_id:
                self.img_id_to_cat_id[img_id] = ann['category_id']

        # 4. Filter relevant IDs
        # If an explicit list of IDs (split) is provided, use it.
        # Otherwise, use all images that have an annotation.
        if image_ids:
            self.valid_ids = [i for i in image_ids if i in self.img_id_to_cat_id]
        else:
            self.valid_ids = list(self.img_id_to_cat_id.keys())
            
        print(f"Dataset initialized: {len(self.valid_ids)} images, {len(self.classes)} classes.")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        
        # Get path
        file_name = self.images[img_id]
        img_path = os.path.join(self.root_dir, file_name)
        
        # Get label (mapped to 0...N-1)
        cct_cat_id = self.img_id_to_cat_id[img_id]
        label = self.cct_id_to_model_idx[cct_cat_id]
        
        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Handle corrupt images gracefully by creating a black image or skipping
            print(f"Warning: Could not open {img_path}. using blank image.")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)
            
        return image, label
