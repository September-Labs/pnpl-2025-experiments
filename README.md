# PNPL 2025 Experiments

Collection of experiments using the PNPL library for MEG/EEG signal processing and classification.

## Experiments

### MEG-DeBERTa Phoneme Classifier
State-of-the-art MEG phoneme classification model featuring:
- DeBERTa-style disentangled attention
- Class-balanced focal loss
- IPA phonetic feature prediction
- Supervised contrastive learning

See [experiments/demega/](experiments/demega/) for details.

### Other Experiments – Non-Demega
Curated catalog of alternative architectures explored during the competition, including convolutional hybrids, multi-scale RNNs, graph models, CTC and LCS variants, and specialist ensembles. Code and configs are documented in [experiments/other_experiments/](experiments/other_experiments/).

## Installation

```bash
git clone https://github.com/September-Labs/pnpl-2025-experiments.git
cd pnpl-2025-experiments
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

Individual experiments may have different licenses. See each experiment's LICENSE file for details.

## Citation

If you use this code in your research, please cite the specific experiment you used.
