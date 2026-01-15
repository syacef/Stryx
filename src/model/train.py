import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from model.vit_student import SSLStudent
from dataset.embedding import EmbeddingDataset
from model.losses import VICRegLoss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparams
    EPOCHS = 30
    MAX_LR = 1e-2
    DINO_DIM = 384
    
    dataset = EmbeddingDataset("./data/embeddings")
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)
    
    model = SSLStudent(input_dim=DINO_DIM, embedding_dim=DINO_DIM).to(device)
    
    # Using AdamW for better regularization with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)

    # Scheduler: Warmup followed by Cosine Annealing
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(loader), 
        epochs=EPOCHS,
        pct_start=0.1
    )
    
    criterion = VICRegLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(device)

            # Forward
            student_emb = model(batch)
            loss = criterion(student_emb, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent VICReg instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step() # Step every batch
            
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    torch.save(model.state_dict(), "student_dino.pth")

