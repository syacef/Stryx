import json
import os
import torch
from torch.utils.data import Dataset

class SafariSupervisedDataset(Dataset):
    def __init__(self, json_path, embeddings_dir):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.embeddings_dir = embeddings_dir
        self.videos = {v['id']: v for v in self.data['videos']}
        
        # 1. Create label mappings for Species and Family
        categories = self.data['categories']
        self.species_map = {c['id']: i for i, c in enumerate(categories)}
        
        # Robustly extract unique families
        raw_families = []
        for c in categories:
            fam = c.get('Family')
            if fam is None or (isinstance(fam, float) and torch.tensor(fam).isnan()):
                raw_families.append('Unknown')
            else:
                raw_families.append(str(fam))

        unique_families = sorted(list(set(raw_families)))
        self.family_map = {name: i for i, name in enumerate(unique_families)}
        
        self.sp_to_fam = {}
        for c in categories:
            fam = c.get('Family')
            fam_name = str(fam) if fam and not (isinstance(fam, float) and torch.tensor(fam).isnan()) else 'Unknown'
            self.sp_to_fam[c['id']] = self.family_map[fam_name]

        self.annotations = []
        for ann in self.data['annotations']:
            if os.path.exists(os.path.join(embeddings_dir, f"sa_fari_{ann['video_id']:06d}.pt")):
                self.annotations.append(ann)
        
        print(f"Dataset loaded: {len(self.annotations)} valid samples.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        emb_path = os.path.join(self.embeddings_dir, f"sa_fari_{ann['video_id']:06d}.pt")
        feature = torch.load(emb_path, map_location='cpu').mean(dim=0)

        species_label = self.species_map[ann['category_id']]
        family_label = self.sp_to_fam[ann['category_id']]
        
        return feature, species_label, family_label
