# Other Experiments – Non-Demega

We tried 58 alternative architectures alongside DeMEGa. These trials informed DeMEGa’s final shape: disentangled attention came after observing Transformers struggle on raw MEG, class-balanced focal and temperature scaling were refined during the LCS/CTC and class-reweighting runs, the IPA multi-task head was inspired by the phonetic feature experiments, multi-scale and patch ideas came from CBraMod/Scales/ViT tests, and sensor priors from graph/dilated models guided how we handle spatial information. Paths and storage details have been anonymized.

## Specialist and Ensemble Approaches
- 13 Phonemes Why – specialist ensemble targeting rare phonemes with parallel branches. Code: [13_phonemes_why.py](architectures/13_phonemes_why.py), [_13_phonemes_why.py](architectures/_13_phonemes_why.py). Config: [13_phonemes_why.yaml](configs/13_phonemes_why.yaml).

## Baselines and Lightweight CNNs
- Baseline – 1x1 Conv1d over 306 channels, flatten, linear head to 39 classes. Code: [baseline.py](architectures/baseline.py). Config: [direct_baseline_config.yaml](configs/direct_baseline_config.yaml).
- Simple CNN – shallow conv stack for phoneme logits. Code: [simple_cnn.py](architectures/simple_cnn.py). Config: [conv_test.yaml](configs/conv_test.yaml).
- Linear MEG – linear projection of flattened windows for a fast reference. Code: [linear_meg.py](architectures/linear_meg.py). Config: [linear_meg.yaml](configs/linear_meg.yaml).

## Convolutional and Temporal Hybrids
- CNN plus LSTM – convolutional feature stem feeding an LSTM for temporal context. Code: [cnn_lstm.py](architectures/cnn_lstm.py). Config: [conv_lstm_config.yaml](configs/conv_lstm_config.yaml).
- Conv Transformer – lightweight conv stem followed by a Transformer encoder. Code: [conv_trans.py](architectures/conv_trans.py). Config: [conv_trans.yaml](configs/conv_trans.yaml).
- Danet Simple – dual-branch conv and attention design with optional pretraining. Code: [danet_simple.py](architectures/danet_simple.py). Config: [danet_simple.yaml](configs/danet_simple.yaml).
- SEANet – convolutional backbone with squeeze-and-excitation style channel attention. Code: [seanet.py](architectures/seanet.py). Config: [seanet.yaml](configs/seanet.yaml).
- Scales – multi-scale conv and attention stack blending coarse and fine temporal cues. Code: [scales.py](architectures/scales.py). Config: [scales.yaml](configs/scales.yaml).
- CBraMod – patch-based spatial and spectral embedding with Transformer encoder and pooled classifier. Code: [cbramod.py](architectures/cbramod.py). Config: [cbramod.yaml](configs/cbramod.yaml).
- CAPE – context-aware phoneme embedding pipeline prior to classification. Code: [cape.py](architectures/cape.py). Config: none available.
- Mistr – residual conv and attention blocks adapted for MEG. Code: [mistr.py](architectures/mistr.py). Config: [mistr_config.yaml](configs/mistr_config.yaml).

## Recurrent and Sequence Models
- GRU Network – stacked GRUs with temporal smoothing for denoising. Code: [gru_network.py](architectures/gru_network.py). Config: [gru_network.yaml](configs/gru_network.yaml).
- Willet Network – five-layer GRU stack with dropout and linear head. Code: [willet_network.py](architectures/willet_network.py). Config: [willet_network.yaml](configs/willet_network.yaml).
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
- YOLO MEG – YOLO backbone repurposed for framewise phoneme scoring. Code: [yolo.py](architectures/yolo.py). Config: [yolo.yaml](configs/yolo.yaml).
- Prototype and metric methods are listed under Hyperdimensional and Metric Methods.

## CTC and LCS Family
- MEG LCS CTC family – LCS-style encoder with CTC loss and Zipf weighting, plus stability-tuned revisions. Code: [meg_lcs_ctc.py](architectures/meg_lcs_ctc.py), [meg_lcs_ctc_v2.py](architectures/meg_lcs_ctc_v2.py), [meg_lcs_ctc_v3.py](architectures/meg_lcs_ctc_v3.py). Configs: [config_meg_lcs_ctc.yaml](configs/config_meg_lcs_ctc.yaml), [config_meg_lcs_ctc_bm.yaml](configs/config_meg_lcs_ctc_bm.yaml), [config_meg_lcs_ctc_v2.yaml](configs/config_meg_lcs_ctc_v2.yaml), [config_meg_lcs_ctc_v2_bm.yaml](configs/config_meg_lcs_ctc_v2_bm.yaml), [config_meg_lcs_ctc_v2_bm2.yaml](configs/config_meg_lcs_ctc_v2_bm2.yaml) and related size and regularization sweeps stored alongside, [meg_lcs_ctc_v3.yaml](configs/meg_lcs_ctc_v3.yaml).
- KAN and spectral variants – KAN layers and spectral decomposition enhancements. Code: [meg_lcs_ctc_kan.py](architectures/meg_lcs_ctc_kan.py), [meg_lcs_ctc_kan_sd.py](architectures/meg_lcs_ctc_kan_sd.py), [meg_lcs_ctc_kan_reg.py](architectures/meg_lcs_ctc_kan_reg.py), [meg_lcs_ctc_ph.py](architectures/meg_lcs_ctc_ph.py), [meg_lcs_ctc_sd.py](architectures/meg_lcs_ctc_sd.py). Configs: [meg_lcs_ctc_kan.yaml](configs/meg_lcs_ctc_kan.yaml), [meg_lcs_ctc_ph.yaml](configs/meg_lcs_ctc_ph.yaml), [meg_lcs_ctc_sd.yaml](configs/meg_lcs_ctc_sd.yaml).
- Two-Step CTC – alignment stage followed by classifier fine-tuning. Code: [meg_two_step_ctc.py](architectures/meg_two_step_ctc.py). Config: [meg_two_step_ctc.yaml](configs/meg_two_step_ctc.yaml).
- Unified LCS – three-stage unified LCS training flow. Code: [unified_lcs.py](architectures/unified_lcs.py). Config: [unified_lcs.yaml](configs/unified_lcs.yaml).
- LCS Phoneme – enhanced LCS-style conv stack tuned for phoneme logits. Code: [lcs_phoneme.py](architectures/lcs_phoneme.py). Config: [lcs_phoneme.yaml](configs/lcs_phoneme.yaml).

## Graph and Spatial Models
- Spatial GNN – graph neural network over MEG sensors with temporal pooling. Code: [spatial_gnn.py](architectures/spatial_gnn.py). Config: [spatial_gnn.yaml](configs/spatial_gnn.yaml).
- Spatial Dilated Attention – dilated and Fourier-inspired spatial attention before classification. Code: [spatial_dilated.py](architectures/spatial_dilated.py). Config: [spatial_dilated.yaml](configs/spatial_dilated.yaml).
- SPD Geometric – manifold learning on covariance matrices. Code: [spd_geometric.py](architectures/spd_geometric.py). Config: [spd_geometric.yaml](configs/spd_geometric.yaml).
- STGM – spatial-temporal graph model with Mamba sequence blocks. Code: [stgm.py](architectures/stgm.py). Config: [stgm.yaml](configs/stgm.yaml).

## Phonetic Feature and Decomposition Models
- Phone Decomposition – predicts articulatory feature vectors then maps to phonemes. Code: [phone_deco.py](architectures/phone_deco.py). Config: [phone_deco.yaml](configs/phone_deco.yaml).
- Phonetic Features Experiment – articulation-supervised variant exploring loss formulations. Code: [phonetic_features_experiment.py](architectures/phonetic_features_experiment.py). Config: [phonetic_features_experiment.yaml](configs/phonetic_features_experiment.yaml).

## Additional Assets
- Uniphy – UniPhyNet variant with streamlined metrics. Code: [uniphy.py](architectures/uniphy.py). Config: [uniphy.yaml](configs/uniphy.yaml).
- Default experiment settings – baseline configuration files shared across runs. Configs: [default.yaml](configs/default.yaml), [config.yaml](configs/config.yaml).
- RQA Conf – recurrence quantification analysis settings. Config: [rqa_conf.yaml](configs/rqa_conf.yaml).

## Holdout F1 Macro (selected submissions)

| submission_id | method_name | score | base_architecture | config_path |
| --- | --- | --- | --- | --- |
| 533318 | meg_lcs_ctc_081125v3_submission | 0.41698 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538670 | demega_090625v12_last | 0.41042 | demega | experiments/demega/configs/demega.yaml |
| 538599 | demega_090625v8 | 0.40933 | demega | experiments/demega/configs/demega.yaml |
| 538698 | ensemble_multiple_090625v0 | 0.40854 | ensemble_general | - |
| 538684 | ensemble_demega_090625v8v12 | 0.40502 | ensemble_demega | experiments/demega/configs/demega.yaml |
| 535964 | two_stage_lcs_ctc_082425v0 | 0.39050 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 533313 | meg_lcs_ctc_081125v0_submission | 0.39006 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 533684 | ensemble_gmean_outer | 0.38553 | ensemble_general | - |
| 533591 | enhanced_meg_lcs_ctc_081225v1 | 0.38335 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534239 | meg_lcs_ctc_081525v7 | 0.37976 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 535205 | scales_082125v0 | 0.37895 | scales | configs/scales.yaml |
| 535029 | unified_lcs_082025v1_s3_last | 0.37660 | unified_lcs | configs/unified_lcs.yaml |
| 535030 | unified_lcs_082025v1_s3_last_stand | 0.37619 | unified_lcs | configs/unified_lcs.yaml |
| 534762 | meg_lcs_ctc_081825v3 | 0.37436 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534195 | meg_lcs_ctc_081525v2 | 0.37291 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 536619 | two_stage_lcs_ctc_082725v3 | 0.37277 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 536014 | two_stage_lcs_ctc_082425ev0v1 | 0.36991 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 534761 | meg_lcs_ctc_081825v2 | 0.36969 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 533453 | meg_lcs_ctc_081125v12_submission | 0.36341 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 533516 | meg_lcs_ctc_081125v17 | 0.35069 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 535028 | unified_lcs_082025v1_s3_best | 0.34833 | unified_lcs | configs/unified_lcs.yaml |
| 535038 | unified_lcs_082025v1_s3_last_stand_v2 | 0.34282 | unified_lcs | configs/unified_lcs.yaml |
| 534546 | linear_meg_081725v1 | 0.34245 | linear_meg | configs/linear_meg.yaml |
| 534229 | meg_lcs_ctc_081525v5 | 0.34214 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 535712 | meg_lcs_ctc_082325v0 | 0.33905 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538588 | demega_090625v6 | 0.33586 | demega | experiments/demega/configs/demega.yaml |
| 538669 | demega_090625v12 | 0.33529 | demega | experiments/demega/configs/demega.yaml |
| 533868 | bm2_ensemble_hmean_outer | 0.33488 | bm_ensemble | configs/config_meg_lcs_ctc_v2_bm2_32.yaml |
| 533862 | meg_lcs_ctc_081225v1_submission | 0.33422 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 533861 | meg_lcs_ctc_081225v1 | 0.33422 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 536725 | two_stage_lcs_ctc_082825v0 | 0.33154 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 536019 | two_stage_lcs_ctc_082425v3_g50 | 0.32847 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 534534 | uniphynet_081725v0 | 0.32788 | uniphynet | configs/uniphy.yaml |
| 536671 | two_stage_lcs_ctc_082725v9 | 0.32640 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 534431 | meg_lcs_ctc_v3_081425v8 | 0.32581 | meg_lcs_ctc_v3 | configs/meg_lcs_ctc_v3.yaml |
| 534042 | meg_lcs_ctc_v3_081425v1_last | 0.32581 | meg_lcs_ctc_v3 | configs/meg_lcs_ctc_v3.yaml |
| 538686 | demega_090625v17_last | 0.31973 | demega | experiments/demega/configs/demega.yaml |
| 538663 | demega_090625v10 | 0.31757 | demega | experiments/demega/configs/demega.yaml |
| 534590 | linear_meg_081825v0 | 0.31647 | linear_meg | configs/linear_meg.yaml |
| 535039 | unified_lcs_082025v1_s3_last_stand_v3 | 0.31564 | unified_lcs | configs/unified_lcs.yaml |
| 534243 | meg_lcs_ctc_081525v8 | 0.31338 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538577 | demega_090625v4 | 0.31173 | demega | experiments/demega/configs/demega.yaml |
| 534517 | uniphynet_081625v6 | 0.31003 | uniphynet | configs/uniphy.yaml |
| 534518 | uniphynet_081625v6 | 0.31003 | uniphynet | configs/uniphy.yaml |
| 536011 | two_stage_lcs_ctc_082425v1 | 0.30829 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 533504 | meg_lcs_ctc_081125v16 | 0.30448 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534410 | meg_lcs_ctc_081625v1 | 0.30285 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534041 | long_train2_std | 0.29699 | training_variants | - |
| 534019 | long_train2 | 0.28807 | training_variants | - |
| 534040 | long_train1_std | 0.28711 | training_variants | - |
| 536015 | two_stage_lcs_ctc_082425v2 | 0.28564 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 534049 | meg_lcs_ctc_081425v0_submission | 0.28539 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538688 | demega_090625v18_last | 0.28477 | demega | experiments/demega/configs/demega.yaml |
| 534015 | bm32_ensemble_hmean | 0.28062 | bm_ensemble | configs/config_meg_lcs_ctc_v2_bm2_32.yaml |
| 538687 | demega_090625v18 | 0.27906 | demega | experiments/demega/configs/demega.yaml |
| 534760 | meg_lcs_ctc_081825v1 | 0.27897 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538860 | demega_090725v3_submission | 0.27697 | demega | experiments/demega/configs/demega.yaml |
| 534001 | meg_lcs_ctc_v3_081425v1 | 0.27648 | meg_lcs_ctc_v3 | configs/meg_lcs_ctc_v3.yaml |
| 538594 | lcs_090625v0 | 0.27554 | lcs | configs/lcs_phoneme.yaml |
| 533562 | meg_lcs_ctc_081125v18 | 0.27417 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534181 | meg_lcs_ctc_081525v1 | 0.26827 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534522 | uniphynet_081625v7 | 0.26559 | uniphynet | configs/uniphy.yaml |
| 534017 | bm_smaller1_long | 0.26155 | bm | configs/config_meg_lcs_ctc_v2_bm2_32.yaml |
| 538857 | demega_090725v0_submission | 0.26120 | demega | experiments/demega/configs/demega.yaml |
| 538859 | demega_090725v2_submission | 0.26098 | demega | experiments/demega/configs/demega.yaml |
| 534011 | meg_lcs_ctc_v3_081425v5 | 0.25866 | meg_lcs_ctc_v3 | configs/meg_lcs_ctc_v3.yaml |
| 534766 | meg_lcs_ctc_081825v4 | 0.25763 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538591 | demega_090625v7 | 0.25742 | demega | experiments/demega/configs/demega.yaml |
| 534264 | meg_lcs_ctc_081525v9 | 0.25575 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 535222 | scales_082125v3 | 0.25380 | scales | configs/scales.yaml |
| 537003 | gru_network_082825v0 | 0.25175 | gru_network | configs/gru_network.yaml |
| 534278 | meg_lcs_ctc_081525v10 | 0.25082 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538796 | demega_090625v16_last | 0.25073 | demega | experiments/demega/configs/demega.yaml |
| 534003 | meg_lcs_ctc_v3_081425v1_last | 0.24628 | meg_lcs_ctc_v3 | configs/meg_lcs_ctc_v3.yaml |
| 534414 | meg_lcs_ctc_081625v3 | 0.24223 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538907 | power_demega_090725v0 | 0.23997 | power_demega | experiments/demega/configs/demega.yaml |
| 534905 | lcs_phoneme_082025v0 | 0.23919 | lcs | configs/lcs_phoneme.yaml |
| 538601 | demega_090625v9 | 0.23567 | demega | experiments/demega/configs/demega.yaml |
| 538603 | lcs_090625v2 | 0.23490 | lcs | configs/lcs_phoneme.yaml |
| 536630 | two_stage_lcs_ctc_082725v4 | 0.23470 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 538665 | demega_090625v11_last | 0.23347 | demega | experiments/demega/configs/demega.yaml |
| 538794 | demega_090625v16 | 0.22937 | demega | experiments/demega/configs/demega.yaml |
| 538685 | demega_090625v17 | 0.22911 | demega | experiments/demega/configs/demega.yaml |
| 534593 | linear_meg_081825v2 | 0.22682 | linear_meg | configs/linear_meg.yaml |
| 536030 | two_stage_lcs_ctc_082425v4_g25 | 0.22460 | two_stage_lcs_ctc | configs/two_stage_lcs_ctc.yaml |
| 534415 | uniphynet_081625v0 | 0.22097 | uniphynet | configs/uniphy.yaml |
| 538564 | demega_090625v1 | 0.21684 | demega | experiments/demega/configs/demega.yaml |
| 534231 | meg_lcs_ctc_081525v6 | 0.21540 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534413 | meg_lcs_ctc_081625v2 | 0.21520 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534018 | bm_smaller1_long | 0.21147 | bm | configs/config_meg_lcs_ctc_v2_bm2_32.yaml |
| 534542 | uniphynet_081725v1 | 0.21118 | uniphynet | configs/uniphy.yaml |
| 534022 | meg_lcs_ctc_v3_081425v6_submission | 0.20095 | meg_lcs_ctc_v3 | configs/meg_lcs_ctc_v3.yaml |
| 534027 | meg_lcs_ctc_v3_081425v8_submission | 0.19540 | meg_lcs_ctc_v3 | configs/meg_lcs_ctc_v3.yaml |
| 538547 | demega_090625v0 | 0.19261 | demega | experiments/demega/configs/demega.yaml |
| 533898 | bm_0_571_n6 | 0.19206 | bm | configs/config_meg_lcs_ctc_v2_bm2_32.yaml |
| 534512 | uniphynet_081625v5_submission | 0.18761 | uniphynet | configs/uniphy.yaml |
| 534591 | linear_meg_081825v1 | 0.18676 | linear_meg | configs/linear_meg.yaml |
| 533900 | bm_0_571_n6_no_std | 0.18031 | bm | configs/config_meg_lcs_ctc_v2_bm2_32.yaml |
| 537913 | willet_network_090325v0 | 0.16703 | willet_network | configs/willet_network.yaml |
| 534432 | uniphynet_081625v3 | 0.15996 | uniphynet | configs/uniphy.yaml |
| 534419 | uniphynet_081625v1 | 0.15995 | uniphynet | configs/uniphy.yaml |
| 534016 | short_tiny1_0_35 | 0.15785 | training_variants | - |
| 534417 | megnet_081625v0 | 0.14485 | megnet | configs/megnet.yaml |
| 534407 | meg_lcs_ctc_081625v0 | 0.13934 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 534422 | uniphynet_081625v2 | 0.13552 | uniphynet | configs/uniphy.yaml |
| 538664 | demega_090625v11 | 0.13387 | demega | experiments/demega/configs/demega.yaml |
| 534052 | meg_lcs_ctc_081425v0_submission_ns | 0.11014 | meg_lcs_ctc | configs/config_meg_lcs_ctc.yaml |
| 538576 | demega_090625v5 | 0.11005 | demega | experiments/demega/configs/demega.yaml |
| 535693 | spd_geometric_082325v0 | 0.10543 | spd_geometric | configs/spd_geometric.yaml |
| 535243 | conv_trans_082125v0 | 0.10269 | conv_trans | configs/conv_trans.yaml |
| 537280 | hierarchical_binary_083125v0 | 0.07919 | hierarchical_binary | configs/hierarchical_binary_v2.yaml |
| 535048 | yolo_meg_phoneme_082025v0 | 0.07754 | yolo | configs/yolo.yaml |
| 534543 | linear_meg_081725v0 | 0.07074 | linear_meg | configs/linear_meg.yaml |
| 535743 | hd_network_082325v0 | 0.06422 | hd_network | configs/hd_network.yaml |
| 534063 | eegnet_081425v0 | 0.03994 | eegnet | configs/eegnet.yaml |
| 538589 | lcs_090625v1 | 0.03350 | lcs | configs/lcs_phoneme.yaml |
| 537059 | cape_submission_20250829_105603 | 0.02798 | cape | - |
| 534520 | dipn_081725v0 | 0.02157 | dipn | configs/dipn.yaml |
| 534416 | spatial_gnn_081625v0 | 0.01772 | spatial_gnn | configs/spatial_gnn.yaml |
| 538858 | demega_090725v1_submission | 0.01648 | demega | experiments/demega/configs/demega.yaml |
| 535731 | prototype_network_082325v2 | 0.01351 | prototype_network | configs/prototype_network.yaml |
| 536800 | rf_082825v0 | 0.00327 | training_variants | - |
| 535691 | spatial_dilated_082325v4 | 0.00312 | spatial_dilated | configs/spatial_dilated.yaml |
| 534526 | uniphynet_081725v0 | 0.00154 | uniphynet | configs/uniphy.yaml |

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
