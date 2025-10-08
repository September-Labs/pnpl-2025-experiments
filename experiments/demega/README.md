# DeMEGa: Disentangled MEG Attention Phoneme Classifier

State-of-the-art MEG phoneme classification using DeBERTa-style attention mechanism and class-balanced focal loss.

**Authors:** Aleksandr Smechov, Ihor Stepanov, Alexander Yavorskyi, and Shivam Chaudhary

## Key Features

- **DeBERTa Attention**: Disentangled self-attention mechanism adapted for MEG signals
- **Class-Balanced Focal Loss**: Handles severely imbalanced phoneme distributions
- **IPA Feature Prediction**: Multi-task learning with phonetic features
- **Supervised Contrastive Learning**: Improved representation learning
- **MEG Conformer Architecture**: Specialized layers for MEG data processing
- **Test-Time Augmentation**: Enhanced inference with temporal and channel augmentation

## Architecture Overview

The model combines several innovations:

1. **MEG-specific preprocessing**: Robust normalization using median and IQR
2. **Conformer blocks** with DeBERTa attention for temporal modeling
3. **Multi-task learning** with IPA phonetic feature prediction
4. **Class-balanced focal loss** for handling 39 imbalanced phoneme classes
5. **Supervised contrastive learning** for better feature representations

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
- **Multiple grouping levels** (5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100 samples)
- **39 ARPABET phonemes** with position encoding

### Loading the Dataset

```python
from huggingface_hub import snapshot_download
from pnpl.datasets import GroupedDataset

local_path = snapshot_download(
    repo_id="wordcab/libribrain-meg-preprocessed",
    repo_type="dataset",                 
    allow_patterns=["data/grouped_100**"],    
    local_dir="data" 
)

# Or load locally after downloading
train_dataset = GroupedDataset(
    preprocessed_path="data/grouped_100/train_grouped.h5",
    load_to_memory=True  # Load entire dataset to RAM for faster training
)
```

### Available Grouping Configurations

| Grouping | Train Size | Validation | Test | Total Size |
|----------|------------|------------|------|------------|
| grouped_5 | 45.6 GB | 425 MB | 456 MB | ~47 GB |
| grouped_10 | 22.8 GB | 213 MB | 228 MB | ~24 GB |
| grouped_20 | 11.4 GB | 106 MB | 114 MB | ~12 GB |
| grouped_50 | 4.6 GB | 37 MB | 42 MB | ~4.7 GB |
| grouped_100 | 2.3 GB | 19 MB | 21 MB | ~2.4 GB |

We recommend `grouped_100` for initial experiments (best speed/accuracy tradeoff).

## Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py \
    --config configs/default_config.yaml \
    --data_path /path/to/your/meg/data

# Or train with HuggingFace dataset
python scripts/train.py \
    --config configs/default_config.yaml \
    --use_huggingface \
    --grouping_level 100
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
python scripts/generate_submission.py \
    --config configs/default_config.yaml \
    --checkpoint path/to/best_checkpoint.ckpt
```

## Model Architecture Details

### DeBERTa Attention
- Disentangled content and position representations
- Content-to-position and position-to-content attention
- Log-scaled position buckets for efficient encoding

### MEG Conformer Layer
- Depthwise separable convolutions
- DeBERTa self-attention
- Feed-forward network with SiLU activation
- Pre-layer normalization for stable training

### Loss Functions
- **Balanced Focal Loss**: Combines focal loss with per-class averaging
- **Supervised Contrastive**: NT-Xent loss for representation learning
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

Model performance on LibriBrain 2025 dataset (to be updated):
- **F1 Macro**: TBD
- **Balanced Accuracy**: TBD

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

- **Aleksandr Smechov**
- **Ihor Stepanov**
- **Alexander Yavorskyi**
- **Shivam Chaudhary**

## Contact

For questions or collaborations, please open an issue on GitHub.