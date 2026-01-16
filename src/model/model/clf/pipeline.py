import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto
from tqdm import tqdm

from model.clf.safari_clf import SafariSpeciesClassifier
from model.clf.safari_clf_v2 import SafariSpeciesClassifier as SafariSpeciesClassifierV2


class ModelType(Enum):
    SINGLE_HEAD = auto()
    SINGLE_HEAD_V2 = auto()


class SafariPipeline:
    def __init__(self, model_type: ModelType, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model_type = model_type

        self.model = self._setup_model().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    def _setup_model(self):
        if self.model_type == ModelType.SINGLE_HEAD:
            return SafariSpeciesClassifier(
                self.config["backbone"],
                self.config["emb_dim"],
                self.config["num_species"],
            )
        elif self.model_type == ModelType.SINGLE_HEAD_V2:
            return SafariSpeciesClassifierV2(
                self.config["backbone"],
                self.config["emb_dim"],
                self.config["num_species"],
            )
        raise ValueError(f"Unsupported Model Type: {self.model_type}")

    def run_epoch(self, loader, scheduler=None, is_train=True, epoch_idx=0):
        self.model.train() if is_train else self.model.eval()

        total_loss, correct, total = 0, 0, 0

        # Setup tqdm progress bar
        mode = "Train" if is_train else "Val"
        pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1} [{mode}]", leave=False)

        context = torch.set_grad_enabled(is_train)
        with context:
            for batch_idx, (features, sp_targets, _) in enumerate(pbar):
                # features: [B, T, C, H, W]
                features = features.to(self.device)
                sp_targets = sp_targets.to(self.device)

                # Forward
                sp_logits = self.model(features)
                loss = F.cross_entropy(sp_logits, sp_targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if scheduler:
                        scheduler.step()

                # Metrics calculation
                total_loss += loss.item()
                _, predicted = torch.max(sp_logits, 1)
                batch_correct = (predicted == sp_targets).sum().item()
                correct += batch_correct
                total += sp_targets.size(0)

                # --- REAL-TIME UPDATE ---
                if is_train:
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{100 * batch_correct / sp_targets.size(0):.1f}%",
                        }
                    )

        avg_loss = total_loss / len(loader)
        avg_acc = 100 * correct / total
        return avg_loss, avg_acc
