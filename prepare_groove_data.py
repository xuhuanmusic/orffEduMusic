# prepare_groove_data.py

import os
import csv
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

from data_utils import midi2matrix, compute_note_density_idx, compute_vel_contour, compute_laidbackness, my_skeletonify, NOTE_DENSITY_CLASSES, VEL_CLASSES

# === Configurations ===
RESOLUTION = 24  # Subdivisions per beat
BAR_LENGTH = 2   # Bars per segment
BEATS_PER_BAR = 4
SEQ_LEN = RESOLUTION * BAR_LENGTH * BEATS_PER_BAR
HOP_LEN = SEQ_LEN // 2

TRAIN_PCT, VALID_PCT = 0.8, 0.1

RAW_MIDI_ROOT = "./grooveMidi"  # path to your Groove MIDI files
OUT_PICKLE = "./preprocessed_groove_data.pkl"

# === Find MIDI Files ===
print("Scanning for Groove MIDI files...")
midi_files = glob(f"{RAW_MIDI_ROOT}/**/*.mid", recursive=True)
midi_files = [f for f in midi_files if "groove" in f.lower() or "Groove" in f]

print(f"Found {len(midi_files)} groove-style MIDI files.")

# === Preprocess ===
data_list = []

for midi_file in tqdm(midi_files):
    out = midi2matrix(midi_file, resolution=RESOLUTION, bar_length=BAR_LENGTH)
    if out is None:
        continue

    note, vel, mt, tempo, midi_f = out

    while note.shape[0] < SEQ_LEN:
        note = np.concatenate([note, note], axis=0)
        vel = np.concatenate([vel, vel], axis=0)
        mt = np.concatenate([mt, mt], axis=0)

    for i in range(0, note.shape[0], HOP_LEN):
        if i + SEQ_LEN > note.shape[0]:
            break

        small_note = note[i: i + SEQ_LEN]
        small_vel = vel[i: i + SEQ_LEN]
        small_mt = mt[i: i + SEQ_LEN]

        n_kicks = np.sum(small_note[:, 0])
        n_snares = np.sum(small_note[:, 1])
        nonzero_inst = np.count_nonzero(np.sum(small_note, axis=0))

        if n_kicks == 0 or n_snares == 0 or nonzero_inst < 3:
            continue

        note_density_idx = compute_note_density_idx(small_note, NOTE_DENSITY_CLASSES)
        vel_contour = compute_vel_contour(small_vel, small_note, VEL_CLASSES)
        time_contour = compute_laidbackness(small_mt, small_note)
        skeleton = my_skeletonify(small_note[np.newaxis, :], small_vel[np.newaxis, :], resolution=RESOLUTION)[0]

        # # -------- NEW  split ----------
        # r = random.random()
        # if r < TRAIN_PCT:
        #     split_flag = "train"
        # elif r < TRAIN_PCT + VALID_PCT:
        #     split_flag = "valid"
        # else:
        #     split_flag = "test"
        # # ----------------------------------------------------
        data_list.append({
            "note": small_note,
            "vel": small_vel,
            "mt": small_mt,
            "tempo": tempo,
            "skeleton": skeleton,
            "note_density_idx": note_density_idx,
            "vel_contour": vel_contour,
            "time_contour": time_contour,
            "midi_file": midi_file
            # "split": split_flag  # 👈 New filed
        })

# === Save ===
with open(OUT_PICKLE, "wb") as f:
    pickle.dump(data_list, f)

print(f"Saved {len(data_list)} segments to {OUT_PICKLE}")
