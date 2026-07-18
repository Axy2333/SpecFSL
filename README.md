# SpecFSL: Few-Shot SERS Spectral Recognition Based on Siamese Network

SpecFSL is a few-shot learning framework built on Siamese networks for SERS spectrum recognition. It derives transferable spectral feature extractors via contrastive pretraining to enable few-shot identification of compounds from SERS spectra.

This repository contains two interactive notebooks: one for contrastive pretraining (`pre-training.ipynb`) and one for downstream inference (`infer.ipynb`).

> **Note:** The core concepts and experimental results of this work will be compiled into a paper focusing on few-shot SERS compound identification. Code readability will be continuously improved in subsequent updates, with planned extensions to support additional spectral measurement modes and recognition tasks for complex mixed samples.

---

## Getting Started

If you aim to reproduce the few-shot identification experiments, **`infer.ipynb` is the recommended entry point**. It provides an end-to-end demonstration of extracting spectral features and completing compound classification with the pretrained encoder.

Before running, ensure the following are in place:

1. **Python environment** — all dependent packages installed (see [Installation](#installation) below)
2. **Pretrained checkpoint** — `2trained_model_final_model.pt`, containing the complete parameters of the trained Siamese network
3. **Test dataset** — a `data/` folder containing categorized SERS spectrum subfolders (`.npy` files)

Once the checkpoint and dataset are ready, run all cells in `infer.ipynb` sequentially to launch the full few-shot recognition pipeline.

---

## Installation

```bash
# Create virtual environment
python -m venv specfsl_env

# Activate (Mac/Linux)
source specfsl_env/bin/activate

# Activate (Windows)
# specfsl_env\Scripts\activate

# Install all dependencies
pip install torch torchvision numpy pandas matplotlib seaborn tqdm scikit-learn jupyter
```

---

## Project Structure

```
project-root/
├── data/                          # SERS spectrum test samples (categorized subfolders)
├── pre-training.ipynb             # Siamese network contrastive pretraining
├── infer.ipynb                    # End-to-end few-shot inference and evaluation
└── 2trained_model_final_model.pt  # Pretrained SpecFSL encoder checkpoint
```

---

## Core Mechanism

The checkpoint `2trained_model_final_model.pt` stores all parameters of the Siamese network, which comprises two functional modules:

**1. Feature backbone (reused during inference)**
- Encoder embedding module: dual-branch linear layers that map single spectral wavelength points to a latent space of dimension `d_model = 4`
- Two stacked convolutional feature extraction blocks paired with max-pooling downsampling
- Fully-connected layer outputting fixed **64-dimensional** spectral embedding vectors

**2. Auxiliary matching head (discarded during inference)**
- Sigmoid binary classification branch used only during pretraining to judge the similarity of spectral pairs; excluded from forward propagation at inference time

During inference, `infer.ipynb` instantiates the identical Siamese network architecture, loads weights from the `.pt` checkpoint, and switches to evaluation mode. Only the `FeatureExtract` function is called; all network parameters remain frozen. The pipeline assigns each query spectrum to a compound category by computing the minimum Euclidean distance between query embeddings and support set embeddings, then outputs overall classification accuracy and a confusion matrix.

---

## Workflow

### Step 1: Pretraining

Run `pre-training.ipynb` to complete Siamese network contrastive pretraining. This generates `2trained_model_final_model.pt` in the project root directory.

### Step 2: Prepare Data

Place all compound subfolders containing `.npy` spectrum files inside the `data/` directory.

### Step 3: Run Inference

```bash
jupyter notebook
```

Open `infer.ipynb` and run all cells from top to bottom. The pipeline executes the following steps:

1. Import all dependencies and global hyperparameters
2. Batch-load all spectral data from the `data/` directory
3. Instantiate the Siamese network, load pretrained weights, and freeze the encoder
4. Split spectra into support and query sets according to preset hyperparameters
5. Extract 64-dimensional feature vectors for all support samples
6. Assign each query spectrum to the nearest support class using minimum Euclidean distance
7. Compute global classification accuracy and plot a confusion matrix heatmap

---

## Adjustable Hyperparameters

```python
nS = 1          # Number of support samples per category — adjust to run different K-shot experiments
nnn = 32        # Total number of compound categories
exclude_num = 1  # Support/query split threshold — must be consistent with nS
max_samples = 50 # Maximum number of spectra loaded per compound category
```

---

## Outputs

| Output | Description |
|--------|-------------|
| **Console logs** | Shape statistics of support/query sets per category; overall Top-1 classification accuracy |
| **Confusion matrix** | 32-class heatmap annotated with per-class recognition accuracy |
| **`all_pic_features`** | 64-dim feature vectors of all support samples |
| **`y_pred` / `y_true`** | Predicted and ground-truth labels — exportable as CSV for offline analysis |

---

## Troubleshooting

**`FileNotFoundError` for `2trained_model_final_model.pt`**
Confirm the checkpoint is placed in the project root directory and the filename contains no typos.

**Tensor shape mismatch**
Ensure the network definition is identical in both notebooks. The input spectrum length is fixed at **3660** and must not be changed arbitrarily.

**CUDA out of memory**
Reduce `max_samples` to limit spectra loaded per class, or switch to CPU by modifying the device setting in the notebook.

**Low classification accuracy**
Check in order: (1) generalization quality of the pretrained encoder, (2) support/query split logic, (3) spectrum channel splitting logic for `r` and `x`, (4) distance computation code.

**Overlapping labels in the confusion matrix**
Increase the figure canvas size or reduce the annotation font size in the plotting cell.

---

## Contact & Citation

Feel free to contact us if you have questions about code logic, network architecture, experimental setup, or suggestions for optimization. Contributions and pull requests are welcome.

**If this code is used in your academic research, please cite the corresponding paper upon its official publication.**
