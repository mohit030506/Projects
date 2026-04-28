import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from speech_dataset import RAVDESSDataset
from mfcc_mlp import MFCCMLP
import torch.nn.functional as F
from tqdm import tqdm
import os

BATCH_SIZE = 64
MAX_EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('🚀  Device →', DEVICE)

full_dataset = RAVDESSDataset(root_dir='data/ravdess',
                              sample_rate=16000,
                              max_len_seconds=4)

train_set, val_set = random_split(
    full_dataset,
    [int(len(full_dataset)*(1-VAL_SPLIT)), int(len(full_dataset)*VAL_SPLIT)],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)   # keep 0 for debugging
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

model = MFCCMLP(n_mfcc=40, hidden=256, n_classes=7).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
patience = 5
no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f'Epoch {epoch:02d}', leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            _, preds = torch.max(logits, 1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

    val_acc = correct / total
    avg_loss = epoch_loss / len(train_set)
    print(f'Epoch {epoch:02d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'audio_net_best.pt')
        no_improve = 0
        print('💾  New best audio model saved')
    else:
        no_improve += 1
        if no_improve >= patience:
            print('⏹️  Early stopping')
            break

print('🏁  Audio training finished – best val acc:', best_val_acc)
