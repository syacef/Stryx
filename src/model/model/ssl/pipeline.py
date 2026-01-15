import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

from dataset.video_extractor import VideoFramesDataset
from dataset.embedding_dataset import EmbeddingDataset
from model.ssl.resnet_ssl import ResNetSSL
from model.losses import VICRegLoss

class SafariPipeline:
    def __init__(self, model_type="dinov2_vits14", base_data_dir="./data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.base_data_dir = base_data_dir
        
        self.embeddings_dir = os.path.join(base_data_dir, "embeddings", model_type)
        self.student_save_path = f"student_{model_type}.pth"
        
        os.makedirs(self.embeddings_dir, exist_ok=True)

    def extract_features(self, input_dir, batch_size=32, num_workers=4, overwrite=False):
        video_folders = sorted(glob.glob(os.path.join(input_dir, "sa_fari_*")))
        print(f"Found {len(video_folders)} folders to process.")

        print(f"🚀 Loading {self.model_type} for extraction...")
        model = torch.hub.load("facebookresearch/dinov2", self.model_type).to(self.device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            for folder_path in tqdm(video_folders, desc="Overall Progress"):
                video_id = os.path.basename(folder_path)
                save_path = os.path.join(self.embeddings_dir, f"{video_id}.pt")

                if os.path.exists(save_path) and not overwrite:
                    continue

                frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
                if not frame_paths:
                    continue

                dataset = VideoFramesDataset(frame_paths, transform=transform)
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
                    embeddings = model(batch)
                    video_features.append(embeddings.cpu())

                if video_features:
                    final_tensor = torch.cat(video_features, dim=0)
                    torch.save(final_tensor, save_path)
        
        del model
        torch.cuda.empty_cache()

    def train_ssl(self, epochs=30, batch_size=512, lr=1e-2):
        
        print(f"🧠 Training SSL Student on features in {self.embeddings_dir}")
        dataset = EmbeddingDataset(self.embeddings_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Dimension mapping
        dim = 384 if "vits14" in self.model_type else 768
        model = ResNetSSL(input_dim=dim, embedding_dim=dim).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(loader),
            epochs=epochs,
            pct_start=0.1,
        )
        criterion = VICRegLoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                student_emb = model(batch)
                loss = criterion(student_emb, batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        torch.save(model.state_dict(), self.student_save_path)
        print(f"✅ Pipeline Complete. Student saved to {self.student_save_path}")
