# DeMEGa: Disentangled MEG Attention Phoneme Classifier

State-of-the-art MEG phoneme classification using DeBERTa-style attention mechanism and class-balanced focal loss.

**Authors:** [Aleksandr Smechov](https://www.linkedin.com/in/aleksandr-smechov/), [Ihor Stepanov](https://www.linkedin.com/in/ihor-knowledgator/), [Alexander Yavorskyi](https://www.linkedin.com/in/oleksandr-yavorskyi/), and [Shivam Chaudhary](https://www.linkedin.com/in/shivam199/)

## About September Labs

We're September Labs, and our mission is to scale non-invasive data collection to unprecedented levels, then use this data to train foundation models that help decode patient speech patterns.

We're a multi-disciplinary group, so our approach to MEG phoneme classification likewise borrowed from our collective experience, including medical NLP, speech recognition, and invasive BCI systems. This approach, combined with tons of trial and error, and parameter tuning, got us around ~60% on the holdout for Phase 2 of the Extended Track.

## Key Features

- **DeBERTa Attention**: Disentangled self-attention mechanism adapted for MEG signals
- **Class-Balanced Focal Loss**: Handles severely imbalanced phoneme distributions
- **IPA Feature Prediction**: Multi-task learning with phonetic features
- **Supervised Contrastive Learning**: Improved representation learning
- **MEG Conformer Architecture**: Specialized layers for MEG data processing
- **Wavelet Denoising**: Advanced signal preprocessing with db4 wavelets
- **Temperature-Scaled Class Reweighting**: Automatic handling of rare phonemes

## Architecture Overview

Our architecture combines six key innovations, developed through two months of experimentation and iterative refinement:

### A) DeBERTa Attention for MEG Signals
We adapted DeBERTa's disentangled attention mechanism to MEG temporal data. Using disentangled attention helps capture temporal dependencies and makes positional information dependent on channel information. So we don't just memorize specific patterns at specific time points. This makes the model more invariant to the time axis.

**Implementation:** [`meg_classifier/models/components/deberta_attention.py`](meg_classifier/models/components/deberta_attention.py)

### B) Class-Balanced Focal Loss with IPA Features
We implemented focal loss with per-class averaging to handle severe phoneme imbalance. We augmented this with IPA (International Phonetic Alphabet) phonetic feature prediction, with 14 articulatory features (consonantal, voicing, place of articulation, etc.) as an auxiliary task. This improved generalization a bit, especially for rare phonemes.

**Implementations:**
- Focal Loss: [`meg_classifier/models/components/losses.py`](meg_classifier/models/components/losses.py) - `focal_loss_mean_over_present_classes`
- IPA Features: [`meg_classifier/models/components/ipa_features.py`](meg_classifier/models/components/ipa_features.py)

### C) Supervised Contrastive Learning
We added a supervised NT-Xent contrastive loss to learn better feature representations. Samples from the same phoneme class are pulled together in embedding space while different classes are pushed apart, improving decision boundaries for confusable phonemes.

**Implementation:** [`meg_classifier/models/components/losses.py`](meg_classifier/models/components/losses.py) - `supervised_nt_xent`

### D) Signal Processing & Normalization
We used median-IQR normalization instead of mean-std to handle MEG outliers and artifacts. For data augmentation, we applied temporal jittering (±8-24ms shifts), channel dropout (5%), and Gaussian noise, then used test-time augmentation (TTA) with weighted averaging. TTA wasn't actually effective!

**Implementation:** [`meg_classifier/models/components/conformer.py`](meg_classifier/models/components/conformer.py) - `normalize` function

### E) Temperature-Scaled Class Reweighting
We computed class weights using exponential frequency-based temperature scaling: `w_c = exp(-T × freq_c)`, clamped to [0.5, 5.0]. This automatically upweights rare phonemes without manual tuning.

**Implementation:** [`meg_classifier/models/meg_classifier.py`](meg_classifier/models/meg_classifier.py) - `BalancedPhonemePretrainer`

### F) Wavelet Denoising
Before feeding the holdout data to the trained model, the signal undergoes wavelet decomposition. For each MEG channel, the signal undergoes wavelet decomposition with the **db4 wavelet** (Daubechies 4) at **decomposition level 3**, breaking the signal into 4 sets of coefficients: 1 approximation and 3 detail levels.

The noise level is estimated using the **Median Absolute Deviation (MAD)** method: `sigma = median(abs(finest_details)) / 0.6745`, where 0.6745 is the scaling factor for Gaussian noise consistency.

The threshold is calculated using the universal threshold formula: `threshold = sigma * sqrt(2 * log(n))`, where n is the signal length (125 timepoints). This threshold is scaled by the `denoise_strength` parameter (default 0.6).

**Soft thresholding** is applied to detail coefficients at all decomposition levels except the approximation coefficients. After thresholding, the signal is reconstructed using `waverec`. The `preserve_scale` flag ensures the denoised signal maintains the original mean and standard deviation.

**Implementation:** Available in the `pnpl` library preprocessing utilities

## Development Approach

To arrive at this architecture, we spent two months throwing whatever we could at the holdout, mostly stuff picked up from our backgrounds, and hardly any MEG or EEG specific architecture (ex. EEGFormer). In the last two weeks, we took the best performing architectures and iterated on them.

## Installation

```bash
# Clone the repository
git clone https://github.com/September-Labs/pnpl-2025-experiments.git
cd pnpl-2025-experiments/experiments/demega

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This model uses the **LibriBrain MEG Preprocessed Dataset** available on HuggingFace:
https://huggingface.co/datasets/wordcab/libribrain-meg-preprocessed

The dataset provides pre-grouped and averaged MEG samples for significantly faster loading:
- **306 MEG channels** (102 magnetometers + 204 gradiometers)
- **250 Hz sampling rate**
- **~52 hours of recordings**
- **Multiple grouping levels** (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100 samples - increments of 5)
- **39 ARPABET phonemes** with position encoding

### External Datasets for Extended Track

Since we focused our attention on the Extended Track, we tried various external datasets until we found some that significantly boosted F1 Macro scores:

1. **DSC_3011085.05_995** - MEG dataset from Radboud University
   - URL: https://data.ru.nl/collections/di/dccn/DSC_3011085.05_995
   - Contains phoneme-related MEG recordings with multiple participants

2. **ds006468** - MEG dataset from NEMAR
   - URL: https://nemar.org/dataexplorer/detail?dataset_id=ds006468
   - Includes German phoneme data

Given the similarity of the tasks in these datasets and multiple participants, we think these datasets (even the one with German phonemes) could be a great validation/test set (or vice versa) to test how robust your architecture is on different participants and across languages.

### Downloading the Dataset

**Important:** This dataset contains binary HDF5 (.h5) files, NOT standard datasets. You cannot use `datasets.load_dataset()` - you must use `snapshot_download()` or our download script.

#### Option 1: Using the Download Script (Recommended)

```bash
# Download grouped_100 (2.4 GB) - fastest option for experiments
python download_data.py --grouping_level 100 --output_dir ./data

# Download grouped_20 (12 GB) - better accuracy
python download_data.py --grouping_level 20 --output_dir ./data
```

#### Option 2: Manual Download with Python

```python
from huggingface_hub import snapshot_download

# Download h5 files for grouped_100
snapshot_download(
    repo_id="wordcab/libribrain-meg-preprocessed",
    repo_type="dataset",
    allow_patterns=["data/grouped_100/**"],
    local_dir="./data"
)
```

#### Directory Structure After Download

```
data/
└── grouped_100/
    ├── train_grouped.h5        # Training data
    ├── validation_grouped.h5   # Validation data
    └── test_grouped.h5          # Test data
```

#### Loading Locally

After downloading, the training script will automatically detect and use local h5 files:

```python
from pnpl.datasets import GroupedDataset

train_dataset = GroupedDataset(
    preprocessed_path="./data/grouped_100/train_grouped.h5",
    load_to_memory=True  # Load entire dataset to RAM for faster training
)
```

### Available Grouping Configurations

The dataset is available in multiple grouping levels (increments of 5 from 5 to 100):

| Grouping | Train Size | Validation | Test | Total Size |
|----------|------------|------------|------|------------|
| grouped_5 | 45.6 GB | 425 MB | 456 MB | ~47 GB |
| grouped_10 | 22.8 GB | 213 MB | 228 MB | ~24 GB |
| grouped_15 | 15.2 GB | 142 MB | 152 MB | ~16 GB |
| grouped_20 | 11.4 GB | 106 MB | 114 MB | ~12 GB |
| grouped_25 | 9.1 GB | 85 MB | 91 MB | ~9.6 GB |
| grouped_30 | 7.6 GB | 71 MB | 76 MB | ~8 GB |
| grouped_35 | 6.5 GB | 61 MB | 65 MB | ~6.9 GB |
| grouped_40 | 5.7 GB | 53 MB | 57 MB | ~6 GB |
| grouped_45 | 5.1 GB | 47 MB | 51 MB | ~5.3 GB |
| grouped_50 | 4.6 GB | 37 MB | 42 MB | ~4.7 GB |
| grouped_55 | 4.1 GB | 39 MB | 41 MB | ~4.4 GB |
| grouped_60 | 3.8 GB | 36 MB | 38 MB | ~4 GB |
| grouped_65 | 3.5 GB | 33 MB | 35 MB | ~3.7 GB |
| grouped_70 | 3.3 GB | 30 MB | 33 MB | ~3.4 GB |
| grouped_75 | 3.0 GB | 28 MB | 30 MB | ~3.2 GB |
| grouped_80 | 2.9 GB | 27 MB | 29 MB | ~3 GB |
| grouped_85 | 2.7 GB | 25 MB | 27 MB | ~2.8 GB |
| grouped_90 | 2.5 GB | 24 MB | 25 MB | ~2.7 GB |
| grouped_95 | 2.4 GB | 22 MB | 24 MB | ~2.5 GB |
| grouped_100 | 2.3 GB | 19 MB | 21 MB | ~2.4 GB |

**Recommendation:** Use `grouped_100` for quick experiments and `grouped_20` or lower for production models.

## Quick Start

### Step 1: Download the Dataset

```bash
# Download preprocessed h5 files (recommended - only needs to be done once)
python download_data.py --grouping_level 100 --output_dir ./data
```

### Step 2: Train the Model

#### Option A: Using Preprocessed H5 Files (Recommended)

```bash
# Train with downloaded h5 files
python scripts/train.py \
    --config configs/default_config.yaml \
    --use_huggingface \
    --grouping_level 100 \
    --local_dir ./data

# The script will automatically use existing h5 files if available
# Add --download to force re-download
```

#### Option B: Using Raw MEG Data (Slower)

```bash
# Train with raw MEG data (requires LibriBrain raw dataset)
python scripts/train.py \
    --config configs/default_config.yaml \
    --data_path /path/to/libribrain/raw/data
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py \
    --config configs/default_config.yaml \
    --checkpoint path/to/best_checkpoint.ckpt
```

### Generate Submission

```bash
# Generate predictions for competition
# Note: Ensure config specifies class: SingleStageMEGClassifier
python scripts/generate_submission.py \
    --config configs/default_config.yaml \
    --checkpoint path/to/best_checkpoint.ckpt
```

## Technical Implementation Details

### DeBERTa Attention
The disentangled attention mechanism separates content and position representations, computing content-to-position (c2p) and position-to-content (p2c) attention scores. Log-scaled position buckets enable efficient encoding of long sequences.

**Code:** [`meg_classifier/models/components/deberta_attention.py`](meg_classifier/models/components/deberta_attention.py)

### MEG Conformer Layer
Combines depthwise separable convolutions with DeBERTa self-attention and feed-forward networks using SiLU activation. Pre-layer normalization ensures stable training.

**Code:** [`meg_classifier/models/components/conformer.py`](meg_classifier/models/components/conformer.py)

### Loss Functions
- **Balanced Focal Loss**: Combines focal loss with per-class averaging ([`losses.py`](meg_classifier/models/components/losses.py))
- **Supervised Contrastive**: NT-Xent loss for representation learning ([`losses.py`](meg_classifier/models/components/losses.py))
- **IPA Feature Loss**: Binary cross-entropy for phonetic features

## Configuration

The model is highly configurable through YAML files. Key parameters:

```yaml
model:
  params:
    hidden_dim: 128           # Hidden dimension
    num_conformers: 4         # Number of Conformer layers
    loss_type: "balanced_focal"
    focal_gamma: 2.0          # Focus on hard examples
    use_contrastive: true     # Enable contrastive learning
    use_ipa_features: true    # Enable IPA prediction
```

## Repository Structure

```
demega/
├── meg_classifier/           # Main package
│   ├── models/              # Model implementation
│   │   ├── meg_classifier.py
│   │   └── components/      # Modular components
│   │       ├── deberta_attention.py
│   │       ├── conformer.py
│   │       ├── losses.py
│   │       └── ipa_features.py
├── configs/                 # Configuration files
├── scripts/                 # Training and evaluation
└── requirements.txt         # Dependencies
```

## Performance

Model performance on LibriBrain 2025 dataset:
- **Validation F1 Macro**: 0.45
- **Holdout F1 Macro**: 0.58

## Model Architecture Details

The complete model architecture consists of the following components:

| Name               | Type                      | Params | Mode  |
|--------------------|---------------------------|--------|-------|
| 0  \| pretrainer         | BalancedPhonemePretrainer | 0      | train |
| 1  \| input_projection   | Sequential                | 245 K  | train |
| 2  \| input_skip         | Conv1d                    | 39.3 K | train |
| 3  \| meg_encoder        | ModuleList                | 764 K  | train |
| 4  \| feature_aggregator | Sequential                | 4.1 M  | train |
| 5  \| classifier         | Linear                    | 5.0 K  | train |
| 6  \| ipa_predictor      | Sequential                | 9.2 K  | train |
| 7  \| projection_head    | Sequential                | 33.0 K | train |
| 8  \| feature_norm       | LayerNorm                 | 256    | train |
| 9  \| train_metric       | MulticlassF1Score         | 0      | train |
| 10 \| val_metric         | MulticlassF1Score         | 0      | train |
| 11 \| test_metric        | MulticlassF1Score         | 0      | train |

**Total Parameters:**
- **5.2M** Trainable params
- **0** Non-trainable params
- **5.2M** Total params
- **20.904 MB** Total estimated model params size
- **135** Modules in train mode
- **0** Modules in eval mode

**Main Model Class:** [`SingleStageMEGClassifier`](meg_classifier/models/meg_classifier.py)

## IPA Features

The model predicts 14 phonetic features for each phoneme:
- Consonantal, Syllabic, Sonorant, Voice
- Nasal, Continuant, Labial, Coronal
- Dorsal, High, Low, Back, Round, Diphthong

These features provide linguistic grounding and improve generalization.

## Technical Details

### Data Processing
- MEG signals: 306 channels (204 gradiometers + 102 magnetometers)
- Sampling rate: 250 Hz
- Time window: 0.0-0.5 seconds (125 time points)
- Signal averaging: 100 samples grouped for better SNR

### Training Strategy
- AdamW optimizer with weight decay
- Cosine learning rate schedule with linear warmup
- Gradient clipping for stability
- Optional Stochastic Weight Averaging (SWA)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{demega_2025,
  title={DeMEGa: Disentangled MEG Attention for Phoneme Classification},
  author={Smechov, Aleksandr and Stepanov, Ihor and Yavorskyi, Alexander and Chaudhary, Shivam},
  year={2025},
  url={https://github.com/September-Labs/pnpl-2025-experiments}
}
```

## License

This project is licensed under CC BY-NC 4.0 - see the LICENSE file for details.

## Acknowledgments

- PNPL library by September Labs for MEG data handling
- LibriBrain 2025 competition organizers
- DeBERTa paper for the attention mechanism inspiration

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Authors

- **[Aleksandr Smechov](https://www.linkedin.com/in/aleksandr-smechov/)**
- **[Ihor Stepanov](https://www.linkedin.com/in/ihor-knowledgator/)**
- **[Alexander Yavorskyi](https://www.linkedin.com/in/oleksandr-yavorskyi/)**
- **[Shivam Chaudhary](https://www.linkedin.com/in/shivam199/)**

## Contact

For questions or collaborations, please open an issue on GitHub.