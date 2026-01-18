import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F

class SafariSupervisedDataset(Dataset):
    def __init__(self, json_path, frames_root, labels_txt, num_frames=8, transform=None):
        with open(labels_txt, "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        
        self.species_to_idx = {name: i for i, name in enumerate(self.classes)}

        with open(json_path, "r") as f:
            metadata = json.load(f)

        self.frames_root = frames_root
        self.num_frames = num_frames
        self.transform = transform
        self.valid_samples = []

        for item in metadata.get("video_np_pairs", []):
            if item.get("num_masklets", 0) > 0:
                video_id = item["video_id"]
                species_name = item["noun_phrase"]
                
                folder_name = f"sa_fari_{video_id:06d}"
                folder_path = os.path.join(self.frames_root, folder_name)

                if os.path.isdir(folder_path):
                    frames = sorted([
                        f for f in os.listdir(folder_path) 
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])
                    
                    if len(frames) >= 1:
                        self.valid_samples.append({
                            "label": self.species_to_idx.get(species_name, -1),
                            "folder_path": folder_path,
                            "frame_names": frames
                        })

        self.valid_samples = [s for s in self.valid_samples if s["label"] != -1]
        print(f"Dataset initialized: {len(self.valid_samples)} valid videos found.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        indices = torch.linspace(
            0, len(sample["frame_names"]) - 1, self.num_frames
        ).long()

        clip = []
        for i in indices:
            frame_path = os.path.join(sample["folder_path"], sample["frame_names"][i])
            img = Image.open(frame_path).convert("RGB")
            
            if not self.transform:
                img = img.resize((224, 224), Image.BILINEAR)
                clip.append(F.to_tensor(img))
            else:
                clip.append(self.transform(img))

        clip = torch.stack(clip)
        
        return clip, sample["label"]
