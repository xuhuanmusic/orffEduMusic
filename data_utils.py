# data_utils.py

import numpy as np
import math
from collections import Counter
import miditoolkit

# === Predefined rhythm feature divisions ===
NOTE_DENSITY_CLASSES = [0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225]
VEL_CLASSES = np.arange(0.2, 1.2, 0.2)

# === Groove MIDI main percussion mapping ===
DRUM_CLASSES = ["kick", "snare", "hihat"]
DRUM_MIDI_MAP = {
    "kick": [35, 36],
    "snare": [38, 40],
    "hihat": [42, 44, 46],
}
MIDI2IDX = {m: i for i, cls in enumerate(DRUM_CLASSES) for m in DRUM_MIDI_MAP[cls]}

# === MIDI parsing functions ===
def midi2matrix(midi_file, resolution, bar_length):
    try:
        midi = miditoolkit.MidiFile(midi_file)
        if not any(tr.is_drum for tr in midi.instruments):
            return None
        inst = [tr for tr in midi.instruments if tr.is_drum][0]
    except:
        return None

    ticks_per_beat = midi.ticks_per_beat
    tempo = midi.tempo_changes[0].tempo if midi.tempo_changes else 120

    total_ticks = max((n.end for n in inst.notes), default=0)
    total_beats = math.ceil(total_ticks / ticks_per_beat)
    total_frames = total_beats * resolution
    ticks_per_subdiv = ticks_per_beat // resolution

    note = np.zeros((total_frames, 3))
    vel = np.zeros((total_frames, 3))
    mt = np.zeros((total_frames, 3))

    for n in inst.notes:
        if n.pitch not in MIDI2IDX:
            continue
        t_idx = round(n.start / ticks_per_subdiv)
        if t_idx >= total_frames:
            continue

        micro = (2 * ((n.start % ticks_per_subdiv) / ticks_per_subdiv)) - 1
        vel_val = n.velocity / 127.0

        idx = MIDI2IDX[n.pitch]
        note[t_idx, idx] = 1
        vel[t_idx, idx] = vel_val
        mt[t_idx, idx] = micro

    if np.sum(note) < 10:
        return None

    return [note, vel, mt, tempo, midi_file]

# === Rhythm density classification ===
def compute_note_density_idx(note, density_classes):
    density = np.count_nonzero(note) / (note.shape[0] * note.shape[1])
    for i, t in enumerate(density_classes):
        if density < t:
            return i
    return len(density_classes)

# === Volume contour calculation ===
def compute_vel_contour(vel, note, vel_classes):
    v = vel[note > 0]
    mean_v = np.mean(v) if len(v) > 0 else 0
    for i, t in enumerate(vel_classes):
        if mean_v < t:
            return i
    return len(vel_classes)

# === Micro-rhythm offset contour ===
def compute_laidbackness(mt, note):
    micro = mt[note > 0]
    return float(np.mean(micro)) if len(micro) > 0 else 0.0

# === Simple rhythm skeleton: Main drum beat extraction ===
def my_skeletonify(note, vel, resolution):
    skel = np.zeros_like(note)
    T = note.shape[1]

    # kick: keep strong hits
    kick_v = vel[:, :, 0]
    skel[:, :, 0] = (kick_v > 0.3).astype(int)

    # snare: mainly on beats 2 and 4
    snare = vel[:, :, 1]
    for t in range(T):
        if t % (resolution * 2) == resolution:
            skel[:, t, 1] = int(snare[:, t] > 0.2)

    # hihat: keep 8th note rhythm
    for t in range(0, T, resolution // 2):
        skel[:, t, 2] = 1

    return skel
