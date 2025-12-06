# Other Experiments

We tried 58+ alternative architectures alongside DeMEGa (many not listed individually here). These trials informed DeMEGa’s final shape: disentangled attention came after observing Transformers struggle on raw MEG, class-balanced focal and temperature scaling were refined during the LCS/CTC and class-reweighting runs, the IPA multi-task head was inspired by the phonetic feature experiments, multi-scale and patch ideas came from CBraMod/Scales/ViT tests, and sensor priors from graph/dilated models guided how we handle spatial information. Paths and storage details have been anonymized.

## Specialist and Ensemble Approaches
- 13 Phonemes Why – specialist ensemble targeting rare phonemes with parallel branches. Code: [13_phonemes_why.py](architectures/13_phonemes_why.py), [_13_phonemes_why.py](architectures/_13_phonemes_why.py). Config: [13_phonemes_why.yaml](configs/13_phonemes_why.yaml).

## Baselines and Lightweight CNNs
- Baseline – 1x1 Conv1d over 306 channels, flatten, linear head to 39 classes. Code: [baseline.py](architectures/baseline.py). Config: [direct_baseline_config.yaml](configs/direct_baseline_config.yaml).
- Simple CNN – shallow conv stack for phoneme logits. Code: [simple_cnn.py](architectures/simple_cnn.py). Config: [conv_test.yaml](configs/conv_test.yaml).
- Linear MEG – linear projection of flattened windows for a fast reference (holdout F1 up to 0.342). Code: [linear_meg.py](architectures/linear_meg.py). Config: [linear_meg.yaml](configs/linear_meg.yaml).

## Convolutional and Temporal Hybrids
- CNN plus LSTM – convolutional feature stem feeding an LSTM for temporal context. Code: [cnn_lstm.py](architectures/cnn_lstm.py). Config: [conv_lstm_config.yaml](configs/conv_lstm_config.yaml).
- Conv Transformer – lightweight conv stem followed by a Transformer encoder (holdout F1 ~0.103). Code: [conv_trans.py](architectures/conv_trans.py). Config: [conv_trans.yaml](configs/conv_trans.yaml).
- Danet Simple – dual-branch conv and attention design with optional pretraining. Code: [danet_simple.py](architectures/danet_simple.py). Config: [danet_simple.yaml](configs/danet_simple.yaml).
- SEANet – convolutional backbone with squeeze-and-excitation style channel attention. Code: [seanet.py](architectures/seanet.py). Config: [seanet.yaml](configs/seanet.yaml).
- Scales – multi-scale conv and attention stack blending coarse and fine temporal cues (holdout F1 up to 0.379). Code: [scales.py](architectures/scales.py). Config: [scales.yaml](configs/scales.yaml).
- CBraMod – patch-based spatial and spectral embedding with Transformer encoder and pooled classifier. Code: [cbramod.py](architectures/cbramod.py). Config: [cbramod.yaml](configs/cbramod.yaml).
- CAPE – context-aware phoneme embedding pipeline prior to classification. Code: [cape.py](architectures/cape.py). Config: none available.
- Mistr – residual conv and attention blocks adapted for MEG. Code: [mistr.py](architectures/mistr.py). Config: [mistr_config.yaml](configs/mistr_config.yaml).

## Recurrent and Sequence Models
- GRU Network – stacked GRUs with temporal smoothing for denoising. Code: [gru_network.py](architectures/gru_network.py). Config: [gru_network.yaml](configs/gru_network.yaml).
- Willet Network – five-layer GRU stack with dropout and linear head (holdout F1 ~0.167). Code: [willet_network.py](architectures/willet_network.py). Config: [willet_network.yaml](configs/willet_network.yaml).
- Multiscale LSTM – LSTM at multiple temporal resolutions with channel attention. Code: [multiscale_lstm.py](architectures/multiscale_lstm.py). Config: [multiscale_lstm.yaml](configs/multiscale_lstm.yaml).
- Multi-Scale RNN family – short and long context fusion through stacked RNNs. Code: [multi_scale_rnn.py](architectures/multi_scale_rnn.py), [multi_scale_rnn_enhanced.py](architectures/multi_scale_rnn_enhanced.py), [multi_scale_rnn_pwl.py](architectures/multi_scale_rnn_pwl.py). Configs: [multi_scale_rnn.yaml](configs/multi_scale_rnn.yaml), [multi_scale_rnn_enhanced.yaml](configs/multi_scale_rnn_enhanced.yaml), [multi_scale_rnn_pwl.yaml](configs/multi_scale_rnn_pwl.yaml).
- Outer Limits – outer-loop Transformer attending over inner embeddings to capture long context. Code: [outer_limits.py](architectures/outer_limits.py). Config: [outer_limits.yaml](configs/outer_limits.yaml).

## Hyperdimensional and Metric Methods
- HD Models – hyperdimensional computing classifiers bundling MEG frames. Code: [hd.py](architectures/hd.py). Config: none provided.
- HD Network – hierarchical dynamic network combining CNN and GRU with HD-inspired outputs. Code: [hd_network.py](architectures/hd_network.py). Config: [hd_network.yaml](configs/hd_network.yaml).
- Prototype Network – metric learning with class prototypes. Code: [prototype_network.py](architectures/prototype_network.py). Config: [prototype_network.yaml](configs/prototype_network.yaml).
- Little Classifiers – independent binary heads per phoneme aggregated into multiclass logits. Code: [little_classifiers.py](architectures/little_classifiers.py). Config: [little_classifiers.yaml](configs/little_classifiers.yaml).

## Tree and Hierarchical Classifiers
- Binary Tree Classifier – hierarchical phoneme splits with conformer-like encoder and binary heads. Code: [binary_tree.py](architectures/binary_tree.py), [binary_tree_v2.py](architectures/binary_tree_v2.py). Configs: [binary_tree.yaml](configs/binary_tree.yaml), [binary_tree_v2.yaml](configs/binary_tree_v2.yaml).
- Hierarchical Binary – consistency-enhanced routing to enforce coherent tree decisions. Code: [hierarchical_binary_v1.py](architectures/hierarchical_binary_v1.py), [hierarchical_binary_v2.py](architectures/hierarchical_binary_v2.py). Configs: [hierarchical_binary_v1.yaml](configs/hierarchical_binary_v1.yaml), [hierarchical_binary_v2.yaml](configs/hierarchical_binary_v2.yaml).
- Two-Stage Hierarchies – staged CTC encoder followed by classifier or spectral-density branch. Code: [two_stage_lcs_ctc.py](architectures/two_stage_lcs_ctc.py), [two_stage_meg_sd.py](architectures/two_stage_meg_sd.py). Configs: [two_stage_lcs_ctc.yaml](configs/two_stage_lcs_ctc.yaml), [two_stage_meg_sd.yaml](configs/two_stage_meg_sd.yaml).

## Transformer and State Space Models
- Meg Transformer Phoneme – Transformer encoder over MEG time steps with positional encoding. Code: [meg_transformer_phoneme.py](architectures/meg_transformer_phoneme.py). Config: [meg_transformer_phoneme.yaml](configs/meg_transformer_phoneme.yaml).
- Mamba SSM – state-space backbone for MEG sequences. Code: [mamba.py](architectures/mamba.py). Config: [mamba.yaml](configs/mamba.yaml).
- Mamba Mixture of Experts – Mamba backbone with expert gating per segment. Code: [mamba_moe.py](architectures/mamba_moe.py). Config: none provided.
- YOLO MEG – YOLO backbone repurposed for framewise phoneme scoring (holdout F1 ~0.078). Code: [yolo.py](architectures/yolo.py). Config: [yolo.yaml](configs/yolo.yaml).
- Prototype and metric methods are listed under Hyperdimensional and Metric Methods.

- CTC and LCS family – LCS-style encoder with CTC loss and Zipf weighting, plus stability-tuned revisions (holdout F1 up to 0.417). Code: [meg_lcs_ctc.py](architectures/meg_lcs_ctc.py), [meg_lcs_ctc_v2.py](architectures/meg_lcs_ctc_v2.py), [meg_lcs_ctc_v3.py](architectures/meg_lcs_ctc_v3.py). Configs: [config_meg_lcs_ctc.yaml](configs/config_meg_lcs_ctc.yaml), [config_meg_lcs_ctc_bm.yaml](configs/config_meg_lcs_ctc_bm.yaml), [config_meg_lcs_ctc_v2.yaml](configs/config_meg_lcs_ctc_v2.yaml), [config_meg_lcs_ctc_v2_bm.yaml](configs/config_meg_lcs_ctc_v2_bm.yaml), [config_meg_lcs_ctc_v2_bm2.yaml](configs/config_meg_lcs_ctc_v2_bm2.yaml) and related size and regularization sweeps stored alongside, [meg_lcs_ctc_v3.yaml](configs/meg_lcs_ctc_v3.yaml).
- KAN and spectral variants – KAN layers and spectral decomposition enhancements. Code: [meg_lcs_ctc_kan.py](architectures/meg_lcs_ctc_kan.py), [meg_lcs_ctc_kan_sd.py](architectures/meg_lcs_ctc_kan_sd.py), [meg_lcs_ctc_kan_reg.py](architectures/meg_lcs_ctc_kan_reg.py), [meg_lcs_ctc_ph.py](architectures/meg_lcs_ctc_ph.py), [meg_lcs_ctc_sd.py](architectures/meg_lcs_ctc_sd.py). Configs: [meg_lcs_ctc_kan.yaml](configs/meg_lcs_ctc_kan.yaml), [meg_lcs_ctc_ph.yaml](configs/meg_lcs_ctc_ph.yaml), [meg_lcs_ctc_sd.yaml](configs/meg_lcs_ctc_sd.yaml).
- Two-Step CTC – alignment stage followed by classifier fine-tuning (holdout F1 up to 0.391). Code: [meg_two_step_ctc.py](architectures/meg_two_step_ctc.py). Config: [meg_two_step_ctc.yaml](configs/meg_two_step_ctc.yaml).
- Unified LCS – three-stage unified LCS training flow (holdout F1 up to 0.377). Code: [unified_lcs.py](architectures/unified_lcs.py). Config: [unified_lcs.yaml](configs/unified_lcs.yaml).
- LCS Phoneme – enhanced LCS-style conv stack tuned for phoneme logits (holdout F1 up to 0.276). Code: [lcs_phoneme.py](architectures/lcs_phoneme.py). Config: [lcs_phoneme.yaml](configs/lcs_phoneme.yaml).

## Graph and Spatial Models
- Spatial GNN – graph neural network over MEG sensors with temporal pooling. Code: [spatial_gnn.py](architectures/spatial_gnn.py). Config: [spatial_gnn.yaml](configs/spatial_gnn.yaml).
- Spatial Dilated Attention – dilated and Fourier-inspired spatial attention before classification (holdout F1 ~0.003). Code: [spatial_dilated.py](architectures/spatial_dilated.py). Config: [spatial_dilated.yaml](configs/spatial_dilated.yaml).
- SPD Geometric – manifold learning on covariance matrices. Code: [spd_geometric.py](architectures/spd_geometric.py). Config: [spd_geometric.yaml](configs/spd_geometric.yaml).
- STGM – spatial-temporal graph model with Mamba sequence blocks. Code: [stgm.py](architectures/stgm.py). Config: [stgm.yaml](configs/stgm.yaml).

## Phonetic Feature and Decomposition Models
- Phone Decomposition – predicts articulatory feature vectors then maps to phonemes. Code: [phone_deco.py](architectures/phone_deco.py). Config: [phone_deco.yaml](configs/phone_deco.yaml).
- Phonetic Features Experiment – articulation-supervised variant exploring loss formulations. Code: [phonetic_features_experiment.py](architectures/phonetic_features_experiment.py). Config: [phonetic_features_experiment.yaml](configs/phonetic_features_experiment.yaml).

## Additional Assets
- Uniphy – UniPhyNet variant with streamlined metrics (holdout F1 up to 0.328). Code: [uniphy.py](architectures/uniphy.py). Config: [uniphy.yaml](configs/uniphy.yaml).
- Default experiment settings – baseline configuration files shared across runs. Configs: [default.yaml](configs/default.yaml), [config.yaml](configs/config.yaml).
- RQA Conf – recurrence quantification analysis settings. Config: [rqa_conf.yaml](configs/rqa_conf.yaml).

## Ensembles and Notes
- General and DeMEGa ensembles – blending multiple submissions (holdout F1 up to 0.409–0.408). Configs align with the constituent models (e.g., DeMEGa config in `experiments/demega/configs/demega.yaml`).

## Recovered Checkpoint Summaries
- ViT windowed classifier – 23 sensors, 4 by 25 patches over 200 samples, embedding 128, depth 4, four heads, MLP ratio two, dropout 0.4; trained with learning rate 1e-5, weight decay 0.1, label smoothing 0.1. Source checkpoint: `vit_072925v3`.
- CBraMod multiscale – scales 200, 400, 800 over an 800-sample window, four patches of size 200, eight heads and two layers with hierarchical coarse-to-fine processing, dropout 0.4, learning rate 5e-4. Source checkpoint: `cbramod_073025v10`.
- MEG LCS CTC conformer – 306 channels, hidden 256, four conformer blocks, vocab 39, learning rate 1e-4, Zipf weighting with alpha 0.99 and boost 0.3, alignment enabled. Source logs: `meg_lcs_ctc_081225v0`.
- Shallow CNN speech detector – conv layers 32 at 5x5, 64 at 3x3, 128 at 3x3 with batch norm, channel-attention MLP, silence branch with 16 filters at 7x7, head 2048 to 256 to 64 to 1. Source checkpoint: `cnn_080125v1`.

## Recovered Checkpoint Summaries
- ViT windowed classifier – 23 sensors, 4 by 25 patches over 200 samples, embedding 128, depth 4, four heads, MLP ratio two, dropout 0.4; trained with learning rate 1e-5, weight decay 0.1, label smoothing 0.1. Source checkpoint: `vit_072925v3`.
- CBraMod multiscale – scales 200, 400, 800 over an 800-sample window, four patches of size 200, eight heads and two layers with hierarchical coarse-to-fine processing, dropout 0.4, learning rate 5e-4. Source checkpoint: `cbramod_073025v10`.
- MEG LCS CTC conformer – 306 channels, hidden 256, four conformer blocks, vocab 39, learning rate 1e-4, Zipf weighting with alpha 0.99 and boost 0.3, alignment enabled. Source logs: `meg_lcs_ctc_081225v0`.
- Shallow CNN speech detector – conv layers 32 at 5x5, 64 at 3x3, 128 at 3x3 with batch norm, channel-attention MLP, silence branch with 16 filters at 7x7, head 2048 to 256 to 64 to 1. Source checkpoint: `cnn_080125v1`.
