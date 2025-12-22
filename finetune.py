# finetune.py  – Transfer‑learning & fine‑tuning script
# --------------------------------------------------------
# 1. Loads a Groove‑MIDI–pretrained MusicRhythmNet checkpoint
# 2. Freezes encoder & decoder layers
# 3. Fine‑tunes classification & regression heads on Yorick Children’s Songs dataset
# --------------------------------------------------------

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse, os
from dataset import GrooveMIDIDataset            # <- use same Dataset class (preprocessed Yorick pkl)
from model   import MusicRhythmNet
from evaluate import evaluate_model

# ------------------------------
# utilities
# ------------------------------

def freeze_layers(model, freeze_encoder=True, freeze_decoder=True):
    """Freeze encoder/decoder params to prevent gradient updates."""
    for name, param in model.named_parameters():
        if freeze_encoder and ("lstm" in name or "encoder" in name):
            param.requires_grad = False
        if freeze_decoder and "reconstruct_head" in name:
            param.requires_grad = False
    return model


def get_dataloaders(pkl_path, batch_size=4):
    full_ds = GrooveMIDIDataset(pkl_path)
    train_sz = int(0.8 * len(full_ds))
    val_sz   = len(full_ds) - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))


# ------------------------------
# main fine‑tune routine
# ------------------------------

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load pretrained model
    model = MusicRhythmNet(input_size=9).to(device)
    model.load_state_dict(torch.load(args.pretrained_chkpt, map_location=device))
    print("Loaded pretrained weights from", args.pretrained_chkpt)

    # 2. Freeze selected layers
    model = freeze_layers(model, freeze_encoder=True, freeze_decoder=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # 3. Data
    train_loader, val_loader = get_dataloaders(args.yorick_pkl, batch_size=args.batch)

    # 4. Loss + Optimizer
    loss_recon  = nn.MSELoss()
    loss_class  = nn.CrossEntropyLoss()
    loss_dev    = nn.MSELoss()
    optimizer   = optim.Adam(trainable_params, lr=args.lr)

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train(); tot = 0
        for batch in train_loader:
            x   = batch["input"].to(device)
            y_s = batch["skeleton"].to(device)
            y_c = batch["rhythm_class"].to(device)
            y_d = batch["rhythm_deviation"].to(device)

            optimizer.zero_grad()
            p_s, p_c, p_d = model(x)

            loss1 = loss_recon(p_s, y_s)
            loss2 = loss_class(p_c, y_c)
            loss3 = loss_dev(p_d, y_d)
            loss  = loss1 + loss2 + loss3
            loss.backward(); optimizer.step()
            tot += loss.item()
        avg = tot / len(train_loader)

        metrics = evaluate_model(model, val_loader, device)
        print(f"E{epoch:02d}  train_loss {avg:.4f} | Val: Acc {metrics['acc']:.3f} F1 {metrics['f1']:.3f} MAE {metrics['mae']:.3f}")

        # optional checkpoint
        if epoch % args.ckpt_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"finetuned_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(" → checkpoint saved to", ckpt_path)


# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine‑tune MusicRhythmNet on Yorick dataset")
    parser.add_argument("--pretrained_chkpt", required=True, help="Path to Groove pretrained .pth")
    parser.add_argument("--yorick_pkl",      required=True, help="Path to Yorick preprocessed .pkl")
    parser.add_argument("--epochs",  type=int, default=15)
    parser.add_argument("--batch",   type=int, default=4)
    parser.add_argument("--lr",      type=float, default=5e-4)
    parser.add_argument("--ckpt_every", type=int, default=5)
    parser.add_argument("--out_dir", default="finetuned_ckpts")
    args = parser.parse_args()

    finetune(args)
