from torch.utils.data import DataLoader
import torch
import os
import glob
from torchvision import transforms
from tqdm import tqdm

from dataset.video_extractor import VideoFramesDataset


class DINOv2Extractor:
    def __init__(self, model_type="dinov2_vits14", device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading model {model_type} on {self.device}...")

        self.model = torch.hub.load("facebookresearch/dinov2", model_type).to(
            self.device
        )
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(
        self, input_dir, output_dir, batch_size=32, num_workers=4, overwrite=False
    ):
        os.makedirs(output_dir, exist_ok=True)
        video_folders = sorted(glob.glob(os.path.join(input_dir, "sa_fari_*")))

        print(f"Found {len(video_folders)} folders to process.")

        with torch.no_grad():
            for folder_path in tqdm(video_folders, desc="Overall Progress"):
                video_id = os.path.basename(folder_path)
                save_path = os.path.join(output_dir, f"{video_id}.pt")

                if os.path.exists(save_path) and not overwrite:
                    continue

                frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
                if not frame_paths:
                    continue

                dataset = VideoFramesDataset(frame_paths, transform=self.transform)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                )

                video_features = []
                for batch in loader:
                    if batch is None:
                        continue

                    batch = batch.to(self.device)
                    embeddings = self.model(batch)
                    video_features.append(embeddings.cpu())

                if video_features:
                    final_tensor = torch.cat(video_features, dim=0)
                    torch.save(final_tensor, save_path)
