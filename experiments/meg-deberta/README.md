# MEG-DeBERTa Phoneme Classifier

State-of-the-art MEG phoneme classification using DeBERTa-style attention mechanism and class-balanced focal loss.

## 🌟 Key Features

- **DeBERTa Attention**: Disentangled self-attention mechanism adapted for MEG signals
- **Class-Balanced Focal Loss**: Handles severely imbalanced phoneme distributions
- **IPA Feature Prediction**: Multi-task learning with phonetic features
- **Supervised Contrastive Learning**: Improved representation learning
- **MEG Conformer Architecture**: Specialized layers for MEG data processing
- **Test-Time Augmentation**: Enhanced inference with temporal and channel augmentation

## 📊 Architecture Overview

The model combines several innovations:

1. **MEG-specific preprocessing**: Robust normalization using median and IQR
2. **Conformer blocks** with DeBERTa attention for temporal modeling
3. **Multi-task learning** with IPA phonetic feature prediction
4. **Class-balanced focal loss** for handling 39 imbalanced phoneme classes
5. **Supervised contrastive learning** for better feature representations

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/September-Labs/pnpl-2025-experiments.git
cd pnpl-2025-experiments/experiments/meg-deberta

# Install dependencies
pip install -r requirements.txt
```

## 📝 Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py \
    --config configs/default_config.yaml \
    --data_path /path/to/your/meg/data
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

## 🏗️ Model Architecture Details

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

## 🔧 Configuration

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

## 📂 Repository Structure

```
meg-deberta/
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

## 🎯 Performance

Model performance on LibriBrain 2025 dataset (to be updated):
- **F1 Macro**: TBD
- **Balanced Accuracy**: TBD

## 📖 IPA Features

The model predicts 14 phonetic features for each phoneme:
- Consonantal, Syllabic, Sonorant, Voice
- Nasal, Continuant, Labial, Coronal
- Dorsal, High, Low, Back, Round, Diphthong

These features provide linguistic grounding and improve generalization.

## 🔬 Technical Details

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

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{meg_deberta_2025,
  title={MEG-DeBERTa: Disentangled Attention for MEG Phoneme Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/September-Labs/pnpl-2025-experiments}
}
```

## 📄 License

This project is licensed under CC BY-NC 4.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- PNPL library by September Labs for MEG data handling
- LibriBrain 2025 competition organizers
- DeBERTa paper for the attention mechanism inspiration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.