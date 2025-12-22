# dataset.py
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset

# class GrooveMIDIDataset(Dataset):
#     """ Read preprocessed_groove_data.pkl, return a dict for each sample """
#     def __init__(self, pkl_file):
#         with open(pkl_file, "rb") as f:
#             raw = pickle.load(f)
#
#         self.data = []
#         for itm in raw:
#             note = itm["note"]          # (T,3)
#             vel  = itm["vel"]
#             mt   = itm["mt"]
#             x = torch.tensor(
#                 np.concatenate([note, vel, mt], axis=-1), dtype=torch.float32
#             )                            # (T,9)
#
#             self.data.append({
#                 "input"        : x,                                   #  [T,9]
#                 "note_density" : torch.tensor(itm["note_density_idx"],dtype=torch.long),
#                 "vel_contour"  : torch.tensor(itm["vel_contour"],     dtype=torch.long),
#                 "time_contour" : torch.tensor(itm["time_contour"],    dtype=torch.float32)
#             })
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]            # Return dict

class GrooveMIDIDataset(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        note = torch.tensor(item["note"], dtype=torch.float32)  # [T, 3]
        vel = torch.tensor(item["vel"], dtype=torch.float32)    # [T, 3]
        mt = torch.tensor(item["mt"], dtype=torch.float32)      # [T, 3]
        skeleton = torch.tensor(item["skeleton"], dtype=torch.float32)  # [T, 3]

        # === Model input concatenation: note + vel + mt → [T, 9]
        x = torch.cat([note, vel, mt], dim=-1)

        # === Add rhythm feature fields ===
        rhythm_class = torch.tensor(item["note_density_idx"], dtype=torch.long)  # Classification task
        rhythm_deviation = torch.tensor(item["time_contour"], dtype=torch.float32)  # Regression task

        return {
            "input": x,
            "note": note,
            "vel": vel,
            "mt": mt,
            "skeleton": skeleton,
            "rhythm_class": rhythm_class,
            "rhythm_deviation": rhythm_deviation,
        }

if __name__ == '__main__':
    full_ds = GrooveMIDIDataset('datasets/preprocessed_groove_data.pkl')
    train_size = int(0.8 * len(full_ds))
    val_size = int(0.1 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, val_size, test_size])

    sample = full_ds[0]

    print(sample.keys())
    print("Note shape:", sample["note"].shape)
    print("Rhythm class:", sample["rhythm_class"])
    print("Rhythm deviation:", sample["rhythm_deviation"])