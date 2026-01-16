import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


class SafariSupervisedDataset(Dataset):
    def __init__(self, json_path, frames_root, num_frames=8, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.frames_root = frames_root
        self.num_frames = num_frames
        self.transform = transform

        categories = self.data["categories"]
        self.species_map = {c["id"]: i for i, c in enumerate(categories)}

        unique_families = sorted(
            list(set([str(c.get("Family", "Unknown")) for c in categories]))
        )
        self.family_map = {name: i for i, name in enumerate(unique_families)}
        self.sp_to_fam = {
            c["id"]: self.family_map[str(c.get("Family", "Unknown"))]
            for c in categories
        }

        self.valid_samples = []
        for ann in self.data["annotations"]:
            video_id = ann["video_id"]
            folder_name = f"sa_fari_{video_id:06d}"
            folder_path = os.path.join(self.frames_root, folder_name)

            if os.path.isdir(folder_path):
                frames = sorted(
                    [
                        f
                        for f in os.listdir(folder_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                if len(frames) >= 1:
                    self.valid_samples.append(
                        {"ann": ann, "folder_path": folder_path, "frame_names": frames}
                    )

        print(f"Dataset initialized: {len(self.valid_samples)} videos found.")

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

            img = img.resize((224, 224), Image.BILINEAR)
            clip.append(img)

        if self.transform:
            clip = torch.stack([self.transform(img) for img in clip])
        else:
            clip = torch.stack([F.to_tensor(img) for img in clip])

        species_label = self.species_map[sample["ann"]["category_id"]]
        family_label = self.sp_to_fam[sample["ann"]["category_id"]]

        return clip, species_label, family_label
