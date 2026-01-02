
# orffEduMusic
[![DOI](https://zenodo.org/badge/1120616547.svg)](https://doi.org/10.5281/zenodo.18016360)

> **Real-time rhythmic feedback for children's Orff music education using deep learning**  
> This repository contains the official implementation of **orffEduMusic**, an AI system designed to detect timing deviations in children’s musical performances (ages 4–8) and provide developmentally appropriate feedback. The model is trained via transfer learning from adult groove data and fine-tuned on child-specific recordings.  

---

## 📋 Description
orffEduMusic is an AI-powered system tailored for children's Orff music education. It analyzes rhythmic patterns in musical performances, detects timing deviations, and generates multi-modal feedback (text, visual, and optional audio) to help children improve their rhythmic skills. The system leverages transfer learning from professional drum groove data and fine-tunes on child-specific recordings to adapt to children's musical characteristics.

---

## 📊 Dataset Information
The project uses two primary datasets:

1. **Children's Song Dataset**  
   - Source: [https://doi.org/10.5281/zenodo.4785015](https://doi.org/10.5281/zenodo.4785015)  
   - Description: A collection of children's musical performances (ages 4–8) with annotated rhythmic patterns, used for fine-tuning the model to child-specific rhythmic characteristics.  

2. **Groove MIDI Dataset**  
   - Source: [https://magenta.withgoogle.com/datasets/groove#download](https://magenta.withgoogle.com/datasets/groove#download)  
   - Description: A large dataset of professional drum grooves, used for pre-training the model on general rhythmic patterns before transfer learning. This dataset contains MIDI files of drum performances across various genres, with annotations for tempo, time signature, and drum types.

---

## 💻 Code Information
### File Structure
- `prepare_groove_data.py`: Preprocesses Groove MIDI files into structured matrices (note presence, velocity, micro-timing) for model training. Handles MIDI parsing, segment extraction, and feature calculation (e.g., note density, velocity contour).  
- `dataset.py`: Defines the `GrooveMIDIDataset` class for loading preprocessed data, formatting inputs (concatenated note, velocity, micro-timing features), and preparing targets (skeleton, rhythm class, deviation).  
- `model.py`: Implements the `MusicRhythmNet` model architecture, a bidirectional LSTM-based network with three output heads:  
  - Skeleton reconstruction (for rhythm structure prediction).  
  - Rhythm density classification (8 classes, from sparse to dense).  
  - Micro-timing deviation regression (predicts tempo deviations).  
- `train.py`: Trains the model on preprocessed Groove MIDI data, including training/validation loops, loss calculation (MSE for reconstruction/regression, cross-entropy for classification), and performance evaluation.  
- `finetune.py`: Fine-tunes the pre-trained model on the Children's Song Dataset with transfer learning. Freezes encoder/decoder layers to preserve general rhythmic knowledge while adapting task-specific heads to child data.  
- `evaluate.py`: Computes key evaluation metrics, including accuracy (classification), F1-score, MAE (deviation regression), and onset precision/recall (skeleton reconstruction).  
- `feedback.py`: Generates multi-modal feedback based on model outputs:  
  - Text feedback (assesses rhythm density and timing stability).  
  - Visual feedback (compares predicted vs. true rhythm skeletons via matplotlib).  
  - Optional audio feedback (text-to-speech via pyttsx3).  
- `data_utils.py`: Utility functions for MIDI processing, including `midi2matrix` (converts MIDI to feature matrices), rhythm feature calculation (density, velocity contour, micro-timing), and skeleton extraction.  

---

## 🚀 Usage Instructions

### 1. Setup Environment
```bash
git clone https://github.com/yourname/orffEduMusic.git
cd orffEduMusic
python -m venv env
source env/bin/activate  # Linux/macOS
# For Windows: env\Scripts\activate

pip install -r requirements.txt
```

### 2. Prepare Datasets
#### Groove MIDI Dataset
1. Download the dataset from [https://magenta.withgoogle.com/datasets/groove#download](https://magenta.withgoogle.com/datasets/groove#download).  
2. Extract files to `./grooveMidi` (or update `RAW_MIDI_ROOT` in `prepare_groove_data.py`).  
3. Preprocess the data to generate feature matrices and save as a pickle file:  
   ```bash
   python prepare_groove_data.py
   ```  
   This generates `preprocessed_groove_data.pkl` (default path) containing segmented note, velocity, micro-timing, and derived features.

#### Children's Song Dataset
1. Download the dataset from [https://doi.org/10.5281/zenodo.4785015](https://doi.org/10.5281/zenodo.4785015).  
2. Preprocess using the same pipeline as Groove MIDI (adjust input/output paths in `prepare_groove_data.py` if needed).  

### 3. Train and Fine-Tune
#### Pre-train on Groove MIDI
```bash
python train.py
```
- Trains `MusicRhythmNet` on the preprocessed Groove MIDI data.  
- Saves model checkpoints and logs training/validation metrics (loss, accuracy, F1-score, MAE).

#### Fine-tune on Children's Song Dataset
```bash
python finetune.py --pretrained_chkpt <path_to_pretrained_model.pth> --yorick_pkl <path_to_children_dataset.pkl>
```
- Fine-tunes the pre-trained model on child-specific data.  
- Freezes LSTM encoder and reconstruction head by default to retain general rhythmic knowledge.  
- Saves fine-tuned checkpoints to `./finetuned_ckpts` (configurable via `--out_dir`).

### 4. Evaluate the Model
Evaluation is integrated into `train.py` and `finetune.py`, with metrics logged after each epoch. To run standalone evaluation:
```python
from evaluate import evaluate_model
from model import MusicRhythmNet
from dataset import GrooveMIDIDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MusicRhythmNet().to(device)
model.load_state_dict(torch.load("path/to/model.pth"))

dataset = GrooveMIDIDataset("path/to/evaluation_data.pkl")
loader = DataLoader(dataset, batch_size=4)
metrics = evaluate_model(model, loader, device)
print(metrics)  # Output: acc, f1, mae, onset_precision, onset_recall
```

### 5. Generate Feedback
Use `feedback.py` to generate multi-modal feedback from model outputs:
```python
from feedback import FeedbackManager
import numpy as np

fb = FeedbackManager()

# Example: Generate text feedback (rhythm class 2, deviation 0.18)
text_feedback = fb.generate_text(rhythm_class=2, rhythm_deviation=0.18)
print(text_feedback)

# Example: Generate audio feedback (requires pyttsx3)
fb.generate_audio(text_feedback)

# Example: Generate visual feedback (compare predicted vs true rhythm skeletons)
pred_skel = np.random.rand(100, 3)  # Replace with model output
true_skel = np.random.rand(100, 3)  # Replace with ground truth
save_path = fb.generate_visual(pred_skel, true_skel, save_path="feedback_visual.png")
print(f"Visual feedback saved to {save_path}")
```

---

## 📋 Requirements
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- Librosa ≥ 0.10  
- NumPy
- SciPy
- scikit-learn  
- Matplotlib 
- miditoolkit
- tqdm 
- pyttsx3 
- torchinfo

Install dependencies via:
```bash
pip install torch librosa numpy scipy scikit-learn matplotlib miditoolkit tqdm pyttsx3 torchinfo
```

---

## 📝 Methodology
1. **Data Preprocessing**:  
   - MIDI files are converted into structured matrices (note presence, velocity, micro-timing) using `midi2matrix` in `data_utils.py`.  
   - Segments of fixed length (2 bars, 4 beats/bar, 24 subdivisions/beat) are extracted.  
   - Derived features include note density (8 classes), velocity contour, and micro-timing deviation.  

2. **Model Architecture**:  
   - `MusicRhythmNet` uses a 2-layer bidirectional LSTM for feature extraction.  
   - Three output heads:  
     - Skeleton reconstruction (MSE loss) to predict core rhythm structure.  
     - Rhythm density classification (cross-entropy loss) for 8 density levels.  
     - Micro-timing deviation regression (MSE loss) to predict tempo stability.  

3. **Training Pipeline**:  
   - Pre-training: Model is trained on Groove MIDI data to learn general rhythmic patterns.  
   - Fine-tuning: Encoder and reconstruction layers are frozen; classification/regression heads are adapted to children’s data to preserve general rhythm knowledge while learning child-specific patterns.  

4. **Feedback Generation**:  
   - Text feedback combines rhythm density assessment (e.g., "well-balanced" or "too dense") and timing stability (e.g., "too fast" or "stable").  
   - Visual feedback overlays predicted vs. true rhythm skeletons (Reds for true, Blues for predicted) to highlight discrepancies.  
   - Audio feedback converts text to speech via pyttsx3 for accessibility.

---

## 📚 Citations
If you use this code or dataset, please cite:
- Our paper: [--TODO--]  
- Groove MIDI Dataset: Magenta Team. "Groove MIDI Dataset." (2019). [https://magenta.withgoogle.com/datasets/groove](https://magenta.withgoogle.com/datasets/groove)  
- Children's Song Dataset: [Refer to the citation provided in the Zenodo repository for https://doi.org/10.5281/zenodo.4785015]

---

## 📜 License & Contribution
- License: [MIT License]  
- Contributions: Please open an issue or submit a pull request for bug fixes or feature enhancements. For major changes, contact the authors first to discuss proposed modifications.
