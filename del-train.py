from utils.data_process import MyDataset
from model.model import SmallMind
from torch.utils.data import DataLoader
from config import SmallMindConfig
import torch
from torch.amp import GradScaler, autocast

# Load dataset
config = SmallMindConfig()
train_dataset = MyDataset('./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl', config)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

# Load model
model = SmallMind(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Model parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f} M")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
scaler = GradScaler('cuda') # Mixed precision scaler

def train(model,optimizer, scheduler, train_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y  = x.to(device), y.to(device)
        
        
        with autocast(device_type = device, dtype=torch.float16):
            logits, loss = model(x, targets = y)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(train_loader)

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                logits, loss = model(x, targets=y)
            val_loss += loss.item()
        return val_loss/len(val_loader)

num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train(model, optimizer, scheduler, train_loader, device)
    val_loss  = eval(model, val_loader, device)
    print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    # Save the model checkpoint
    torch.save(checkpoint, f'./checkpoints/model_epoch_{epoch}.pt')
