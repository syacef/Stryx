import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image


class DistillationDataset(Dataset):
    def __init__(self, embeddings_dir, video_base_dir="./data", transform=None):
        self.transform = transform
        self.samples = []

        emb_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.pt")))

        print(f"🔍 Found {len(emb_files)} embedding files. Mapping to frames...")

        for emb_path in emb_files:
            video_id = os.path.basename(emb_path).replace(".pt", "")
            video_folder = os.path.join(video_base_dir, video_id)

            if not os.path.exists(video_folder):
                print(f"⚠️ Warning: Folder {video_folder} not found. Skipping.")
                continue

            features = torch.load(emb_path, map_location="cpu")
            frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
            num_pairs = min(len(frame_paths), len(features))
            for i in range(num_pairs):
                self.samples.append(
                    {"image_path": frame_paths[i], "target_emb": features[i]}
                )

        print(f"✅ Dataset initialized with {len(self.samples)} image-embedding pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample["target_emb"]
