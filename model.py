import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicRhythmNet(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_classes=8):
        super(MusicRhythmNet, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)

        # Rhythm skeleton prediction (reconstruction task)
        self.reconstruct_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Output skeleton dimension: [T, 3]
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

        # Rhythm density classification branch
        self.class_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Classification
        )

        # Micro-rhythm deviation regression branch
        self.deviation_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Regression
        )

    def forward(self, x):  # x shape: [B, T, 9] (concatenated note + vel + mt)
        lstm_out, _ = self.lstm(x)  # [B, T, 2*hidden]
        last_hidden = lstm_out[:, -1, :]  # [B, 2*hidden]

        recon = self.reconstruct_head(lstm_out)  # [B, T, 3]
        cls_out = self.class_head(last_hidden)   # [B, num_classes]
        dev_out = self.deviation_head(last_hidden).squeeze(-1)  # [B]

        return recon, cls_out, dev_out
