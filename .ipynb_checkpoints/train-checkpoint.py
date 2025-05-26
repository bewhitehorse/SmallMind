from utils.data_process import MyDataset
from model.model import SmallMind
from torch.utils.data import DataLoader
from config import SmallMindConfig
import torch

#load dataset
config = SmallMindConfig()
train_dataset = MyDataset('autodl-tmp/dataset/mobvoi_seq_monkey_general_open_corpus.jsonl', config)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[0.9, 0.1])
#define dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size = 12,
    shuffle = True
)
val_loader = DataLoader(
    val_dataset,
    batch_size = 12,
    shuffle = False
)
#load model
model = SmallMind(SmallMindConfig())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 打印模型一共有多少参数

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6} M")
# set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# 设置 cosine 学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

def train(model,optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y  = x.to(device), y.to(device)

        # 前向传播
        logits, loss = model(x, targets = y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        scheduler.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
        return val_loss

for epoch in range(10):
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
    val_loss  = eval(model, val_loader, device)
    print(f"Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    avg_val_loss = val_loss / len(val_loader)

    #save model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
    }
    # Save the model checkpoint
    torch.save(checkpoint, f'autodl-tmp/checkpoints/model_epoch_{epoch}.pt')
