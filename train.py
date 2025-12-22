import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import GrooveMIDIDataset
from model import MusicRhythmNet

from evaluate import evaluate_model

from torchinfo import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

# 1. Load dataset
dataset = GrooveMIDIDataset("datasets/preprocessed_groove_data.pkl")
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# 2. Model and loss functions
model = MusicRhythmNet().to(DEVICE)

recon_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()
dev_loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 3. Training function
def train_epoch(model, loader):
    model.train()
    total_loss, total_recon, total_class, total_dev = 0, 0, 0, 0
    for batch in loader:
        inputs = batch["input"].to(DEVICE)
        targets = batch["skeleton"].to(DEVICE)
        rhythm_class = batch["rhythm_class"].to(DEVICE)
        rhythm_dev = batch["rhythm_deviation"].to(DEVICE)

        optimizer.zero_grad()
        out_recon, out_class, out_dev = model(inputs)
        summary(model, input_size=inputs.shape)

        loss_recon = recon_loss_fn(out_recon, targets)
        loss_class = class_loss_fn(out_class, rhythm_class)
        loss_dev = dev_loss_fn(out_dev, rhythm_dev)

        loss = loss_recon + loss_class + loss_dev
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_class += loss_class.item()
        total_dev += loss_dev.item()

    return total_loss / len(loader), total_recon / len(loader), total_class / len(loader), total_dev / len(loader)

# 4. Validation function
def evaluate(model, loader):
    model.eval()
    total_loss, total_recon, total_class, total_dev = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(DEVICE)
            targets = batch["skeleton"].to(DEVICE)
            rhythm_class = batch["rhythm_class"].to(DEVICE)
            rhythm_dev = batch["rhythm_deviation"].to(DEVICE)

            out_recon, out_class, out_dev = model(inputs)

            loss_recon = recon_loss_fn(out_recon, targets)
            loss_class = class_loss_fn(out_class, rhythm_class)
            loss_dev = dev_loss_fn(out_dev, rhythm_dev)

            loss = loss_recon + loss_class + loss_dev

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_class += loss_class.item()
            total_dev += loss_dev.item()

    return total_loss / len(loader), total_recon / len(loader), total_class / len(loader), total_dev / len(loader)

# 5. Training loop
for epoch in range(EPOCHS):
    train_total, train_recon, train_class, train_dev = train_epoch(model, train_loader)
    val_total, val_recon, val_class, val_dev = evaluate(model, val_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f" Train | Total: {train_total:.4f} | Recon: {train_recon:.4f} | Class: {train_class:.4f} | Dev: {train_dev:.4f}")
    print(f"  Val  | Total: {val_total:.4f} | Recon: {val_recon:.4f} | Class: {val_class:.4f} | Dev: {val_dev:.4f}")


    metrics = evaluate_model(model, val_loader, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Acc {metrics['acc']:.3f} | F1 {metrics['f1']:.3f} | "
          f"MAE {metrics['mae']:.3f} | "
          f"P {metrics['onset_precision']:.3f} | R {metrics['onset_recall']:.3f}")
