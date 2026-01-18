"""
SSL Model Evaluation Script

This script evaluates the trained SSL student model by:
1. Computing feature quality metrics (cosine similarity with teacher)
2. Testing on downstream classification task
3. Visualizing learned representations
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model.ssl.tinyvit import DistilledTinyViT
from model.ssl.pipeline import SSLPipeline, StudentType, TeacherType
from dataset.supervised_dataset import SafariSupervisedDataset
from dataset.distillation_dataset import DistillationDataset


def evaluate_ssl_quality(model, teacher_model, val_loader, device):
    """
    Evaluate SSL model quality by comparing student embeddings with teacher embeddings
    """
    model.eval()
    teacher_model.eval()
    
    cosine_similarities = []
    mse_errors = []
    
    with torch.no_grad():
        for images, teacher_embs in val_loader:
            images = images.to(device)
            teacher_embs = teacher_embs.to(device)
            
            # Get student embeddings
            student_embs = model(images)
            
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                student_embs, teacher_embs, dim=1
            )
            cosine_similarities.extend(cos_sim.cpu().numpy())
            
            # Compute MSE
            mse = torch.mean((student_embs - teacher_embs) ** 2, dim=1)
            mse_errors.extend(mse.cpu().numpy())
    
    return {
        "avg_cosine_similarity": np.mean(cosine_similarities),
        "std_cosine_similarity": np.std(cosine_similarities),
        "avg_mse": np.mean(mse_errors),
        "std_mse": np.std(mse_errors),
    }


def extract_embeddings(model, val_loader, device):
    """Extract embeddings from the SSL model"""
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, teacher_embs in val_loader:
            images = images.to(device)
            student_embs = model(images)
            embeddings.append(student_embs.cpu().numpy())
    
    if embeddings:
        return np.vstack(embeddings)
    return np.array([])


def visualize_embeddings(embeddings, method="pca", output_path="embedding_viz.png"):
    """
    Visualize learned embeddings using PCA or t-SNE
    """
    if len(embeddings) < 2:
        print("Not enough embeddings to visualize")
        return
    
    # Reduce to 2D
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=30)
    plt.title(f"SSL Embeddings Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"Saved visualization to {output_path}")
    plt.close()


def evaluate_on_downstream_task(ssl_model, test_loader, device, classifier_checkpoint="weights/linear_probe_best.pth", num_classes=91, use_new_model=True):
    """
    Evaluate SSL model on downstream classification task.
    Supports both old (safari_clf_epoch15.pth) and new (linear_probe_best.pth) models.
    """
    ssl_model.eval()
    
    # Check if using new linear probe model or old model
    if use_new_model and "linear_probe" in classifier_checkpoint:
        # New model architecture (simpler)
        print(f"Loading new linear probe classifier from {classifier_checkpoint}...")
        
        class LinearProbeClassifier(nn.Module):
            def __init__(self, embedding_dim=384, num_classes=91):
                super().__init__()
                self.refiner = nn.Sequential(
                    nn.Linear(embedding_dim, 512),
                    nn.BatchNorm1d(512, track_running_stats=False),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                )
                self.classifier = nn.Linear(512, num_classes)
            
            def forward(self, x):
                x = self.refiner(x)
                return self.classifier(x)
        
        full_classifier = LinearProbeClassifier(384, num_classes).to(device)
        
        if os.path.exists(classifier_checkpoint):
            full_classifier.load_state_dict(torch.load(classifier_checkpoint, map_location=device))
            print(f"✓ Successfully loaded linear probe classifier")
            full_classifier.eval()
        else:
            print(f"✗ Classifier checkpoint not found: {classifier_checkpoint}")
            return {"downstream_accuracy": 0.0, "downstream_f1": 0.0}
        
        # Evaluate with new model
        print("\nEvaluating linear probe classifier...")
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, _, labels = batch

                images = images.to(device)
                labels = labels.cpu().numpy()
                
                # Handle 5D (video) or 4D (image) inputs
                if images.dim() == 5:
                    b, t, c, h, w = images.shape
                    images_reshaped = images.view(b * t, c, h, w)
                    embeddings_all = ssl_model(images_reshaped)
                    embeddings = embeddings_all.view(b, t, -1).mean(dim=1)
                else:
                    embeddings = ssl_model(images)

                logits = full_classifier(embeddings)
                preds = logits.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        return {
            "downstream_accuracy": accuracy,
            "downstream_f1": f1,
        }
    
    # Old model loading (original code)
    # The trained classifier has: SSL(384) -> Refiner(384->512) -> Head(512->91)
    # We need to reconstruct both the refiner and the head
    # Note: Using individual layers to properly load BatchNorm
    refiner_linear = nn.Linear(384, 512).to(device)
    refiner_bn = nn.BatchNorm1d(512, track_running_stats=False).to(device)  # No running stats in checkpoint
    refiner_act = nn.SiLU().to(device)
    refiner_dropout = nn.Dropout(0.2).to(device)
    
    classifier_head = nn.Linear(512, num_classes).to(device)
    
    # Load pre-trained weights
    print(f"Loading pre-trained classifier from {classifier_checkpoint}...")
    if os.path.exists(classifier_checkpoint):
        checkpoint = torch.load(classifier_checkpoint, map_location=device)
        
        # Extract refiner weights (needs careful mapping for individual layers)
        refiner_state_linear = {}
        refiner_state_bn = {}
        head_state = {}
        
        for k, v in checkpoint.items():
            if k == "refiner.0.weight":
                refiner_state_linear["weight"] = v
            elif k == "refiner.0.bias":
                refiner_state_linear["bias"] = v
            elif k == "refiner.1.weight":
                refiner_state_bn["weight"] = v
            elif k == "refiner.1.bias":
                refiner_state_bn["bias"] = v
            # Note: running_mean/running_var not in checkpoint
            elif k == "species_head.weight":
                head_state["weight"] = v
            elif k == "species_head.bias":
                head_state["bias"] = v
        
        if refiner_state_linear and refiner_state_bn and head_state:
            try:
                refiner_linear.load_state_dict(refiner_state_linear)
                refiner_bn.load_state_dict(refiner_state_bn, strict=False)  # Allow missing running stats
                classifier_head.load_state_dict(head_state)
                print(f"✓ Successfully loaded refiner and classifier head")
                print(f"  - Refiner: 384 -> 512")
                print(f"  - Head: 512 -> {num_classes}")
            except Exception as e:
                print(f"✗ Error loading weights: {e}")
                return {"downstream_accuracy": 0.0, "downstream_f1": 0.0}
        else:
            print("✗ Could not find required keys in checkpoint.")
            return {"downstream_accuracy": 0.0, "downstream_f1": 0.0}
        
    else:
        print(f"✗ Classifier checkpoint not found: {classifier_checkpoint}")
        return {"downstream_accuracy": 0.0, "downstream_f1": 0.0}
    
    refiner_linear.eval()
    refiner_bn.eval()
    refiner_act.eval()
    refiner_dropout.eval()
    classifier_head.eval()
    
    # Evaluate
    print("\nEvaluating pre-trained classifier...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, _, labels = batch

            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Handle 5D (video) or 4D (image) inputs
            if images.dim() == 5:
                b, t, c, h, w = images.shape
                images_reshaped = images.view(b * t, c, h, w)
                embeddings_all = ssl_model(images_reshaped)
                embeddings = embeddings_all.view(b, t, -1).mean(dim=1)
            else:
                embeddings = ssl_model(images)

            # Pass through refiner then classifier head
            refined = refiner_linear(embeddings)
            refined = refiner_bn(refined)
            refined = refiner_act(refined)
            refined = refiner_dropout(refined)
            logits = classifier_head(refined)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return {
        "downstream_accuracy": accuracy,
        "downstream_f1": f1,
    }


if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEACHER_TYPE = TeacherType.DINOV2_VITS14.value
    STUDENT_TYPE = StudentType.TINYVIT.value
    STUDENT_CHECKPOINT = "weights/student_epoch24_tinyvit.pth"  # Update to your latest epoch
    DATA_DIR = "./data"
    BATCH_SIZE = 32
    
    print("=" * 80)
    print("SSL MODEL EVALUATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Teacher: {TEACHER_TYPE}")
    print(f"Student: {STUDENT_TYPE}")
    print(f"Student Checkpoint: {STUDENT_CHECKPOINT}\n")
    
    # 1. Load SSL model
    print("[1/4] Loading SSL student model...")
    ssl_model = DistilledTinyViT(teacher_dim=384).to(DEVICE)
    
    if os.path.exists(STUDENT_CHECKPOINT):
        ssl_model.load_state_dict(torch.load(STUDENT_CHECKPOINT, map_location=DEVICE))
        print(f"✓ Loaded checkpoint: {STUDENT_CHECKPOINT}")
    else:
        print(f"✗ Checkpoint not found: {STUDENT_CHECKPOINT}")
        print("Available checkpoints:")
        for f in sorted(glob.glob("student_epoch*.pth")):
            print(f"  - {f}")
        exit(1)
    
    # 2. Evaluate SSL quality (if teacher is available)
    print("\n[2/4] Evaluating SSL quality on validation set...")
    try:
        teacher_model = torch.hub.load(
            "facebookresearch/dinov2", TEACHER_TYPE
        ).to(DEVICE)
        
        # Create validation dataset (use first 10% of training data for validation)
        dataset = DistillationDataset(
            embeddings_dir=os.path.join(DATA_DIR, "embeddings", TEACHER_TYPE),
            video_base_dir=DATA_DIR,
            transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]),
        )
        
        val_size = len(dataset) // 10
        val_dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), len(dataset) // val_size)[:val_size])
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
        
        quality_metrics = evaluate_ssl_quality(ssl_model, teacher_model, val_loader, DEVICE)
        
        print(f"  Avg Cosine Similarity: {quality_metrics['avg_cosine_similarity']:.4f} ± {quality_metrics['std_cosine_similarity']:.4f}")
        print(f"  Avg MSE: {quality_metrics['avg_mse']:.4f} ± {quality_metrics['std_mse']:.4f}")
        
        del teacher_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ Could not evaluate SSL quality: {e}")
    
    # 3. Visualize embeddings
    print("\n[3/4] Extracting and visualizing embeddings...")
    try:
        dataset = DistillationDataset(
            embeddings_dir=os.path.join(DATA_DIR, "embeddings", TEACHER_TYPE),
            video_base_dir=DATA_DIR,
            transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]),
        )
        
        # Sample subset for visualization
        sample_size = min(1000, len(dataset))
        sample_dataset = torch.utils.data.Subset(
            dataset, np.random.choice(len(dataset), sample_size, replace=False)
        )
        sample_loader = DataLoader(sample_dataset, batch_size=BATCH_SIZE, num_workers=4)
        
        embeddings = extract_embeddings(ssl_model, sample_loader, DEVICE)
        print(f"  Extracted {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        
        visualize_embeddings(embeddings, method="pca", output_path="ssl_embeddings_pca.png")
        
    except Exception as e:
        print(f"  ✗ Could not visualize embeddings: {e}")
    
    # 4. Evaluate on downstream task
    print("\n[4/4] Evaluating on downstream species classification...")
    try:
        # Load supervised dataset with proper transforms (matching SSL training)
        downstream_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        downstream_dataset = SafariSupervisedDataset(
            "./data/sa_fari_train_ext.json",
            "./data",
            "./data/labels.txt",
            transform=downstream_transform
        )
        
        test_size = len(downstream_dataset) // 5
        test_dataset = torch.utils.data.Subset(
            downstream_dataset, range(0, len(downstream_dataset), len(downstream_dataset) // test_size)[:test_size]
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
        
        # Try to use new linear probe model if available, otherwise fall back to old model
        linear_probe_path = "weights/linear_probe_best.pth"
        if os.path.exists(linear_probe_path):
            downstream_metrics = evaluate_on_downstream_task(
                ssl_model, test_loader, DEVICE, 
                classifier_checkpoint=linear_probe_path,
                num_classes=91,
                use_new_model=True
            )
        else:
            print("  Note: Linear probe model not found, using safari_clf_epoch15.pth")
            downstream_metrics = evaluate_on_downstream_task(
                ssl_model, test_loader, DEVICE,
                classifier_checkpoint="weights/safari_clf_epoch15.pth",
                num_classes=91,
                use_new_model=False
            )
        
        print(f"  Downstream Accuracy: {downstream_metrics['downstream_accuracy']:.4f}")
        print(f"  Downstream F1-Score: {downstream_metrics['downstream_f1']:.4f}")
        
    except Exception as e:
        print(f"  ✗ Could not evaluate on downstream task: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nResults Summary:")
    try:
        print(f"  • SSL Quality - Cosine Similarity: {quality_metrics['avg_cosine_similarity']:.4f}")
        print(f"  • Visualization saved: ssl_embeddings_pca.png")
        print(f"  • Downstream Task - Accuracy: {downstream_metrics['downstream_accuracy']:.4f}")
    except:
        print("  • Check output above for individual metrics")

