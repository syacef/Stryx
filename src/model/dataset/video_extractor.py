from torch.utils.data import Dataset
from PIL import Image


class VideoFramesDataset(Dataset):
    def __init__(self, frame_paths, transform=None):
        self.frame_paths = frame_paths
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
