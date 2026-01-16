from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import torch

from model.clf.pipeline import SafariPipeline, ModelType
from model.ssl.tinyvit import DistilledTinyViT
from dataset.supervised_dataset import SafariSupervisedDataset

if __name__ == "__main__":
    full_dataset = SafariSupervisedDataset(
        "./data/sa_fari_train_ext.json", 
        "./data", 
        "./data/labels.txt"
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Split dataset: {len(train_ds)} training samples, {len(val_ds)} validation samples.")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

    num_species = 91 

    state_dict = torch.load(
        "weights/student_epoch22_tinyvit.pth",
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    backbone_model = DistilledTinyViT()
    backbone_model.load_state_dict(state_dict)
    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.eval()

    config = {
        "backbone": backbone_model,
        "emb_dim": 384,
        "num_species": num_species,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "epochs": 20,
    }

    pipeline = SafariPipeline(ModelType.SINGLE_HEAD, config)

    scheduler = OneCycleLR(
        pipeline.optimizer,
        max_lr=config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
    )

    best_val_loss = float("inf")
    epochs = config["epochs"]

    print(f"\n🚀 Starting Training on {pipeline.device}")
    print(f"{'Epoch':<8} | {'T-Loss':<8} | {'T-Acc':<10} | {'V-Loss':<8} | {'V-Acc':<10} | {'Status'}")
    print("-" * 75)

    for epoch in range(epochs):
        train_loss, train_acc = pipeline.run_epoch(
            train_loader, scheduler=scheduler, is_train=True, epoch_idx=epoch
        )

        val_loss, val_acc = pipeline.run_epoch(
            val_loader, is_train=False, epoch_idx=epoch
        )

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(pipeline.model.state_dict(), "best_safari_model.pth")
            status = "Best"

        if (epoch + 1) % 5 == 0:
            torch.save(pipeline.model.state_dict(), f"safari_clf_epoch{epoch+1}.pth")

        print(f"{epoch+1:<8} | {train_loss:<8.4f} | {train_acc:<9.2f}% | {val_loss:<8.4f} | {val_acc:<9.2f}% | {status}")

    print("-" * 75)
    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.4f}")
