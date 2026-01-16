import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from enum import Enum

from dataset.video_extractor import VideoFramesDataset
from dataset.embedding_dataset import EmbeddingDataset
from dataset.distillation_dataset import DistillationDataset
from model.ssl.resnet_ssl import ResNetSSL
from model.ssl.tinyvit import DistilledTinyViT
from model.losses import VICRegLoss

class StudentType(Enum):
    TINYVIT = "tinyvit"
    RESNET_SSL = "resnet_ssl"

class TeacherType(Enum):
    DINOV2_VITS14 = "dinov2_vits14"
    DINOV2_VITB14 = "dinov2_vitb14"

class SSLPipeline:
    def __init__(self, teacher_type=TeacherType.DINOV2_VITS14.value, student_type=StudentType.TINYVIT.value, base_data_dir="./data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_type = teacher_type
        self.student_type = student_type
        self.base_data_dir = base_data_dir

        self.embeddings_dir = os.path.join(base_data_dir, "embeddings", teacher_type)
        self.student_save_path = f"student_{student_type}.pth"

        os.makedirs(self.embeddings_dir, exist_ok=True)

    def extract_features(self, input_dir, batch_size=16, num_workers=4, overwrite=False):
        video_folders = sorted(glob.glob(os.path.join(input_dir, "sa_fari_*")))
        print(f"Found {len(video_folders)} folders to process.")

        print(f"Loading {self.teacher_type} for extraction...")
        model = torch.hub.load("facebookresearch/dinov2", self.teacher_type).to(self.device)
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

    def train_ssl(self, video_input_dir, epochs=30, batch_size=32, lr=1e-3, start_epoch=None):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Training SSL Student: {self.student_type} from Teacher: {self.teacher_type}")
        dataset = DistillationDataset(
            embeddings_dir=self.embeddings_dir, 
            video_base_dir=video_input_dir, 
            transform=train_transform
        )
        dim = 384 if "vits14" in self.teacher_type else 768

        model = None
        if self.student_type == StudentType.TINYVIT.value:
            model = DistilledTinyViT(teacher_dim=dim).to(self.device)
        else:
            model = ResNetSSL(input_dim=3, embedding_dim=dim).to(self.device)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs
        )

        mse_loss = nn.MSELoss()

        if start_epoch is not None:
            model.load_state_dict(torch.load(f"student_epoch{start_epoch}_{self.student_type}.pth"))
            print(f"Resumed training from epoch {start_epoch}")

        for epoch in range(start_epoch+1 if start_epoch is not None else 0, epochs):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, teacher_embs in pbar:
                images = images.to(self.device)        # [Batch, 3, 224, 224]
                teacher_embs = teacher_embs.to(self.device) # [Batch, 384]

                # Student processes the image
                student_embs = model(images) 

                # Combine MSE and Cosine for better feature alignment
                loss_mse = mse_loss(student_embs, teacher_embs)
                loss_cos = 1 - torch.nn.functional.cosine_similarity(student_embs, teacher_embs).mean()
                loss = loss_mse + loss_cos

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            print(f"Finish Epoch {epoch+1} | Avg Loss: {epoch_loss/len(loader):.4f}")
            torch.save(model.state_dict(), f"student_epoch{epoch+1}_{self.model_type}.pth")

        torch.save(model.state_dict(), self.student_save_path)
        print(f"Pipeline Complete. Student saved to {self.student_save_path}")

        del model
        torch.cuda.empty_cache()
