"""
PH-CTC: Persistent Homology CTC for MEG-based Phoneme Classification
Replaces LCS alignment with topological tracking using persistent homology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score
from collections import defaultdict
import gudhi  # For persistent homology computation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ============================================
# Persistent Homology Feature Extraction
# ============================================

class PersistenceComputer:
    """
    Efficient computation of persistence diagrams from MEG data.
    Uses Gudhi for topological feature extraction.
    """
    
    def __init__(self, max_dimension=2, max_edge_length=10.0, resolution=50):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.resolution = resolution
    
    def compute_persistence(self, meg_window: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute persistence diagram for a MEG window.
        
        Args:
            meg_window: (channels, time_points) MEG data
            
        Returns:
            Dictionary mapping dimension -> persistence pairs
        """
        # Build Rips complex from MEG channels as points in time-series space
        rips_complex = gudhi.RipsComplex(points=meg_window.T, max_edge_length=self.max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
        
        # Compute persistence
        persistence = simplex_tree.compute_persistence()
        
        # Extract persistence diagrams by dimension
        diagrams = {}
        for dim in range(self.max_dimension + 1):
            pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            if len(pairs) > 0:
                # Filter infinite persistence
                pairs = pairs[pairs[:, 1] < np.inf]
                if len(pairs) > 0:
                    diagrams[dim] = pairs
                else:
                    diagrams[dim] = np.array([[0, 0]])
            else:
                diagrams[dim] = np.array([[0, 0]])
        
        return diagrams
    
    def persistence_image(self, diagram: np.ndarray, sigma=0.5) -> np.ndarray:
        """
        Convert persistence diagram to persistence image for differentiability.
        
        Args:
            diagram: (n_points, 2) array of birth-death pairs
            sigma: Gaussian kernel bandwidth
            
        Returns:
            (resolution, resolution) persistence image
        """
        if len(diagram) == 0:
            return np.zeros((self.resolution, self.resolution))
        
        # Create grid
        x_min, x_max = 0, self.max_edge_length
        y_min, y_max = 0, self.max_edge_length
        x_grid = np.linspace(x_min, x_max, self.resolution)
        y_grid = np.linspace(y_min, y_max, self.resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Weight function (persistence)
        weights = diagram[:, 1] - diagram[:, 0]
        
        # Compute persistence image
        image = np.zeros((self.resolution, self.resolution))
        for i, (birth, death) in enumerate(diagram):
            if death > birth:  # Valid persistence pair
                gaussian = np.exp(-((X - birth)**2 + (Y - death)**2) / (2 * sigma**2))
                image += weights[i] * gaussian
        
        return image / (image.max() + 1e-8)

class TopologicalFeatureExtractor(nn.Module):
    """
    Learnable topological feature extraction from MEG data.
    Combines persistence computation with neural network processing.
    """
    
    def __init__(self, meg_channels=306, hidden_dim=256, max_dimension=2, 
                 window_size=25, stride=5, resolution=32):
        super().__init__()
        self.meg_channels = meg_channels
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.stride = stride
        self.resolution = resolution
        self.max_dimension = max_dimension  # Store this
        
        # Persistence computer
        self.persistence_computer = PersistenceComputer(
            max_dimension=max_dimension,
            resolution=resolution
        )
        
        # Learnable persistence image parameters
        self.sigma = nn.Parameter(torch.tensor(0.5))
        self.filtration_scale = nn.Parameter(torch.ones(meg_channels))
        
        # Calculate dimension for each persistence dim feature
        # Make sure total adds up to hidden_dim
        dim_per_persistence = hidden_dim // (max_dimension + 1)
        remainder = hidden_dim % (max_dimension + 1)
        
        # CNN for persistence image processing (per dimension)
        self.persistence_cnn = nn.ModuleDict()
        for dim in range(max_dimension + 1):
            # Add remainder to the last dimension to ensure exact match
            output_dim = dim_per_persistence + (remainder if dim == max_dimension else 0)
            self.persistence_cnn[str(dim)] = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, output_dim)  # Exact dimension
            )
        
        # Combine features from different dimensions - now input is exactly hidden_dim
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal attention for combining windows
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

    def extract_windows(self, meg_data: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract overlapping windows from MEG data.
        
        Args:
            meg_data: (B, channels, time_points)
            
        Returns:
            List of (B, channels, window_size) tensors
        """
        B, C, T = meg_data.shape
        windows = []
        
        for start in range(0, T - self.window_size + 1, self.stride):
            window = meg_data[:, :, start:start + self.window_size]
            windows.append(window)
        
        return windows
    
    def forward(self, meg_data: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Extract topological features from MEG data.
        
        Args:
            meg_data: (B, channels, time_points)
            
        Returns:
            features: (B, n_windows, hidden_dim) topological features
            diagrams: List of persistence diagrams for visualization
        """
        B = meg_data.size(0)
        device = meg_data.device
        
        # Apply learnable filtration scaling
        scaled_meg = meg_data * self.filtration_scale.view(1, -1, 1)
        
        # Extract windows
        windows = self.extract_windows(scaled_meg)
        n_windows = len(windows)
        
        if n_windows == 0:  # Handle edge case where sequence is too short
            # Return zero features
            return torch.zeros(B, 1, self.hidden_dim, device=device), []
        
        all_features = []
        all_diagrams = []
        
        for window in windows:
            window_features = []
            window_diagrams = []
            
            for b in range(B):
                # Convert to numpy for persistence computation
                meg_np = window[b].cpu().detach().numpy()
                
                # Compute persistence diagrams
                diagrams = self.persistence_computer.compute_persistence(meg_np)
                window_diagrams.append(diagrams)
                
                # Convert to persistence images and process
                dim_features = []
                for dim in range(self.max_dimension + 1):  # Use stored max_dimension
                    if dim in diagrams:
                        # Create persistence image
                        pers_img = self.persistence_computer.persistence_image(
                            diagrams[dim], 
                            sigma=self.sigma.item()
                        )
                        
                        # Convert to tensor and process with CNN
                        pers_tensor = torch.from_numpy(pers_img).float().to(device)
                        pers_tensor = pers_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                        
                        # Extract features
                        features = self.persistence_cnn[str(dim)](pers_tensor)
                        dim_features.append(features.squeeze(0))
                    else:
                        # No features in this dimension - use correct output size
                        dim_per_persistence = self.hidden_dim // (self.max_dimension + 1)
                        remainder = self.hidden_dim % (self.max_dimension + 1)
                        output_dim = dim_per_persistence + (remainder if dim == self.max_dimension else 0)
                        dim_features.append(torch.zeros(output_dim, device=device))
                
                # Combine features from all dimensions - should now be exactly hidden_dim
                combined = torch.cat(dim_features, dim=0)
                window_features.append(combined)
            
            # Stack batch
            window_features = torch.stack(window_features)  # (B, hidden_dim)
            all_features.append(window_features)
            all_diagrams.append(window_diagrams)
        
        # Stack all windows
        all_features = torch.stack(all_features, dim=1)  # (B, n_windows, hidden_dim)
        
        # Apply feature combiner
        all_features = self.feature_combiner(all_features)
        
        # Apply temporal attention
        all_features, _ = self.temporal_attention(all_features, all_features, all_features)
        
        return all_features, all_diagrams
# ============================================
# Topological Cost Matrix Learner
# ============================================

class TopologicalCostLearner(nn.Module):
    """
    Learn cost matrix based on topological similarity between MEG windows and phoneme templates.
    Replaces MEGCostMatrixLearner with persistence-based matching.
    """
    
    def __init__(self, vocab_size=39, meg_channels=306, hidden_dim=256, 
                 projection_dim=32, n_templates=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_templates = n_templates
        
        # Topological feature extractor
        self.topo_extractor = TopologicalFeatureExtractor(
            meg_channels=meg_channels,
            hidden_dim=hidden_dim
        )
        
        # Learn phoneme topological templates
        self.phoneme_templates = nn.Parameter(
            torch.randn(vocab_size, n_templates, hidden_dim)
        )
        
        # Wasserstein distance approximation network
        self.wasserstein_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for template selection
        self.template_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        
        # Output projection
        self.cost_projection = nn.Linear(hidden_dim, projection_dim)
        self.label_embedding = nn.Embedding(vocab_size, projection_dim)
    
    def compute_topological_distance(self, features1: torch.Tensor, 
                                    features2: torch.Tensor) -> torch.Tensor:
        """
        Compute approximate Wasserstein distance between topological features.
        
        Args:
            features1: (B, N, D) first set of features
            features2: (B, M, D) second set of features
            
        Returns:
            (B, N, M) distance matrix
        """
        B, N, D = features1.shape
        M = features2.shape[1]
        
        # Expand for pairwise computation
        f1_exp = features1.unsqueeze(2).expand(B, N, M, D)
        f2_exp = features2.unsqueeze(1).expand(B, N, M, D)
        
        # Concatenate and compute distance
        combined = torch.cat([f1_exp, f2_exp], dim=-1)
        distances = self.wasserstein_net(combined).squeeze(-1)
        
        return distances
    
    def forward(self, meg_data: torch.Tensor, text_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute topological cost matrix.
        
        Args:
            meg_data: (B, channels, time_points)
            text_labels: (B, L) phoneme labels
            
        Returns:
            (B, L, T) cost matrix
        """
        B, L = text_labels.shape
        
        # Extract topological features from MEG
        meg_features, _ = self.topo_extractor(meg_data)  # (B, n_windows, hidden_dim)
        n_windows = meg_features.shape[1]
        
        # Get phoneme templates for the labels
        label_templates = []
        for b in range(B):
            batch_templates = []
            for l in range(L):
                phoneme_id = text_labels[b, l]
                templates = self.phoneme_templates[phoneme_id]  # (n_templates, hidden_dim)
                
                # Use attention to select relevant template
                query = meg_features[b].unsqueeze(0)  # (1, n_windows, hidden_dim)
                key = value = templates.unsqueeze(0)  # (1, n_templates, hidden_dim)
                attended, _ = self.template_attention(query, key, value)
                batch_templates.append(attended.squeeze(0).mean(0))  # Average over windows
            
            label_templates.append(torch.stack(batch_templates))
        
        label_templates = torch.stack(label_templates)  # (B, L, hidden_dim)
        
        # Compute topological distances
        distances = self.compute_topological_distance(
            label_templates,  # (B, L, hidden_dim)
            meg_features      # (B, n_windows, hidden_dim)
        )  # (B, L, n_windows)
        
        # Convert distances to costs (lower distance = lower cost)
        cost_matrix = 1.0 - distances
        cost_matrix = F.softmax(cost_matrix, dim=1)
        
        return cost_matrix

# ============================================
# PH-enhanced Zipf Weight Learner
# ============================================

class TopologicalZipfLearner(nn.Module):
    """
    Enhanced Zipf learner that incorporates topological complexity.
    Extends ZipfWeightLearner with persistence-based features.
    """
    
    def __init__(self, vocab_size: int, meg_dim: int, hidden_dim: int = 256, alpha: float = 0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        
        # Reuse base Zipf statistics tracking
        self.register_buffer('phoneme_counts', torch.ones(vocab_size))
        self.register_buffer('total_count', torch.tensor(vocab_size, dtype=torch.float32))
        
        # Topological complexity per phoneme
        self.register_buffer('topo_complexity', torch.ones(vocab_size))
        self.register_buffer('topo_stability', torch.ones(vocab_size))
        
        # Learn prototypical topological features per phoneme
        self.register_buffer('topo_prototypes', torch.zeros(vocab_size, hidden_dim))
        
        # Learnable parameters
        self.zipf_s = nn.Parameter(torch.tensor(1.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.complexity_weight = nn.Parameter(torch.tensor(0.3))
        
        # Networks for combining Zipf and topological information
        self.zipf_topo_combiner = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for zipf weight, complexity, stability
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Softmax(dim=-1)
        )
    
    def update_topological_stats(self, phonemes: torch.Tensor, 
                                topo_features: torch.Tensor,
                                persistence_values: List[float]):
        """
        Update topological statistics for phonemes.
        
        Args:
            phonemes: (B,) phoneme labels
            topo_features: (B, hidden_dim) topological features
            persistence_values: List of total persistence values
        """
        with torch.no_grad():
            for i, phoneme in enumerate(phonemes):
                # Update topological prototype
                old_prototype = self.topo_prototypes[phoneme]
                self.topo_prototypes[phoneme] = (
                    self.alpha * old_prototype + (1 - self.alpha) * topo_features[i]
                )
                
                # Update complexity (total persistence)
                if i < len(persistence_values):
                    self.topo_complexity[phoneme] = (
                        self.alpha * self.topo_complexity[phoneme] + 
                        (1 - self.alpha) * persistence_values[i]
                    )
                
                # Update stability (inverse of variance in persistence)
                # This is simplified - in practice you'd track variance over time
                self.topo_stability[phoneme] *= self.alpha
                self.topo_stability[phoneme] += (1 - self.alpha)
    
    def forward(self, topo_features: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Compute Zipf weights enhanced with topological information.
        
        Args:
            topo_features: (B, hidden_dim) topological features
            training: Whether in training mode
            
        Returns:
            (B, vocab_size) weight adjustments
        """
        B = topo_features.size(0)
        device = topo_features.device
        
        # Get base Zipf weights
        frequencies = self.phoneme_counts / self.total_count
        sorted_freqs, sorted_indices = torch.sort(frequencies, descending=True)
        ranks = torch.zeros_like(frequencies)
        ranks[sorted_indices] = torch.arange(1, self.vocab_size + 1, dtype=torch.float32, device=device)
        zipf_weights = 1.0 / torch.pow(ranks, self.zipf_s)
        zipf_weights = zipf_weights / zipf_weights.sum()
        
        # Normalize topological complexity
        norm_complexity = self.topo_complexity / (self.topo_complexity.sum() + 1e-8)
        norm_stability = self.topo_stability / (self.topo_stability.sum() + 1e-8)
        
        # Compute similarity to topological prototypes for each batch item
        similarities = F.cosine_similarity(
            topo_features.unsqueeze(1),  # (B, 1, hidden_dim)
            self.topo_prototypes.unsqueeze(0),  # (1, vocab_size, hidden_dim)
            dim=2
        )  # (B, vocab_size)
        
        # Expand Zipf weights, complexity, and stability for batch
        zipf_weights_batch = zipf_weights.unsqueeze(0).expand(B, -1)  # (B, vocab_size)
        complexity_batch = norm_complexity.unsqueeze(0).expand(B, -1)  # (B, vocab_size)
        stability_batch = norm_stability.unsqueeze(0).expand(B, -1)  # (B, vocab_size)
        
        # Combine all information using weighted average
        weights = (
            zipf_weights_batch * 0.4 +
            similarities * 0.3 +
            complexity_batch * 0.15 +
            stability_batch * 0.15
        )
        
        # Apply temperature and normalize
        weights = F.softmax(weights / self.temperature, dim=-1)
        
        return weights

# ============================================
# Main PH-CTC Model
# ============================================

class PHCTC(L.LightningModule):
    """
    Persistent Homology CTC model for MEG phoneme classification.
    Replaces LCS alignment with topological tracking.
    """
    
    def __init__(self,
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=256,
                 num_conformers=4,
                 learning_rate=1e-4,
                 use_alignment=True,
                 use_topo_zipf=True,
                 zipf_alpha=0.99,
                 zipf_boost_factor=0.3,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # Topological feature extractor
        self.topo_extractor = TopologicalFeatureExtractor(
            meg_channels=meg_channels,
            hidden_dim=hidden_dim
        )
        
        # Reuse MEGConformerLayer from original code
        self.conformers = nn.ModuleList([
            MEGConformerLayer(hidden_dim, 4, hidden_dim*2) 
            for _ in range(num_conformers)
        ])
        
        # CTC components
        self.ctc_projection = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for blank
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size)
        )
        
        # Topological cost learner (replaces MEGCostMatrixLearner)
        self.use_alignment = use_alignment
        if use_alignment:
            self.cost_learner = TopologicalCostLearner(
                vocab_size, meg_channels, hidden_dim
            )
        
        # Topological Zipf learner
        self.use_topo_zipf = use_topo_zipf
        if use_topo_zipf:
            self.topo_zipf_learner = TopologicalZipfLearner(
                vocab_size, hidden_dim, hidden_dim, alpha=zipf_alpha
            )
            self.zipf_boost_factor = zipf_boost_factor
        
        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
    
    def compute_total_persistence(self, diagrams: List[Dict]) -> List[float]:
        """
        Compute total persistence for a batch of persistence diagrams.
        
        Args:
            diagrams: List of persistence diagram dictionaries
            
        Returns:
            List of total persistence values
        """
        persistence_values = []
        for diagram_dict in diagrams:
            total_persistence = 0
            for dim, pairs in diagram_dict.items():
                if isinstance(pairs, np.ndarray) and len(pairs) > 0:
                    persistence = pairs[:, 1] - pairs[:, 0]
                    total_persistence += persistence.sum()
            persistence_values.append(total_persistence)
        return persistence_values
    
    def forward(self, x, use_ctc=False):
        B, C, T = x.shape
        
        # Extract topological features
        topo_features, diagrams = self.topo_extractor(x)  # (B, n_windows, hidden_dim)
        
        # Pool over windows for classification
        features = topo_features.mean(dim=1)  # (B, hidden_dim)
        
        # Apply conformers for temporal modeling
        topo_enhanced = topo_features
        for conformer in self.conformers:
            topo_enhanced = conformer(topo_enhanced)
        
        if use_ctc:
            # CTC path with topological features
            logits = self.ctc_projection(topo_enhanced)  # (B, n_windows, vocab_size+1)
            return logits
        else:
            # Standard classification path
            # Use both pooled topological features and conformer output
            combined_features = torch.cat([
                features,  # Pooled topological
                topo_enhanced.mean(dim=1)  # Pooled conformer output
            ], dim=-1)
            
            # Project to correct dimension for classifier
            combined_features = combined_features[:, :self.classifier[0].in_features]
            
            logits = self.classifier(combined_features)  # (B, vocab_size)
            
            # Apply topological Zipf weighting if enabled
            if self.use_topo_zipf and not self.training:
                zipf_adjustments = self.topo_zipf_learner(features, training=False)
                probs = F.softmax(logits, dim=-1)
                adjusted_probs = (1 - self.zipf_boost_factor) * probs + self.zipf_boost_factor * zipf_adjustments
                logits = torch.log(adjusted_probs + 1e-10)
            
            return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Standard classification
        y_hat = self(x, use_ctc=False)
        loss = self.criterion(y_hat, y)
        
        # Extract topological features for Zipf update
        if self.use_topo_zipf:
            with torch.no_grad():
                topo_features, diagrams = self.topo_extractor(x)
                pooled_features = topo_features.mean(dim=1)
                
                # Compute persistence values
                persistence_values = []
                for b in range(x.size(0)):
                    # Get the first window's diagram for simplicity
                    if len(diagrams) > 0 and len(diagrams[0]) > b:
                        persistence_values.append(
                            self.compute_total_persistence([diagrams[0][b]])[0]
                        )
                    else:
                        persistence_values.append(0.0)
                
                # Update topological statistics
                self.topo_zipf_learner.update_topological_stats(
                    y, pooled_features, persistence_values
                )
        
        # Optional: Add topological CTC loss
        if self.use_alignment and batch_idx % 10 == 0:
            ctc_logits = self(x, use_ctc=True)
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full((x.size(0),), ctc_logits.size(1), dtype=torch.long)
            target_lengths = torch.ones(x.size(0), dtype=torch.long)
            ctc_targets = y.unsqueeze(1)
            
            ctc_loss = self.ctc_loss(log_probs, ctc_targets, input_lengths, target_lengths)
            loss = 0.7 * loss + 0.3 * ctc_loss
            
            self.log('train_ctc_loss', ctc_loss, prog_bar=False)
        
        f1_macro = self.f1_macro(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        # Log topological statistics
        if self.use_topo_zipf and batch_idx % 100 == 0:
            self.log('topo_complexity_mean', self.topo_zipf_learner.topo_complexity.mean())
            self.log('topo_stability_mean', self.topo_zipf_learner.topo_stability.mean())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, use_ctc=False)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Separate parameter groups
        params = [
            {'params': self.topo_extractor.parameters(), 'lr': self.hparams.learning_rate * 0.5},
            {'params': self.conformers.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate},
        ]
        
        if self.use_alignment:
            params.append({'params': self.cost_learner.parameters(), 'lr': self.hparams.learning_rate * 0.5})
        
        if self.use_topo_zipf:
            params.append({'params': self.topo_zipf_learner.parameters(), 'lr': self.hparams.learning_rate * 0.1})
        
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

# ============================================
# Integration and Usage
# ============================================

def create_ph_ctc_model(dataset_info, use_topo_zipf=True):
    """
    Create PH-CTC model for LibriBrain dataset.
    
    Args:
        dataset_info: Dictionary with dataset information
        use_topo_zipf: Whether to enable topological Zipf weighting
    """
    model = PHCTC(
        meg_channels=dataset_info.get('meg_channels', 306),
        time_points=dataset_info.get('time_points', 125),
        vocab_size=dataset_info.get('num_phonemes', 39),
        hidden_dim=256,
        num_conformers=4,
        learning_rate=1e-4,
        use_alignment=True,
        use_topo_zipf=use_topo_zipf,
        zipf_alpha=0.99,
        zipf_boost_factor=0.3
    )
    
    return model

# ============================================
# MEG-adapted Conformer Layer
# ============================================

class MEGConformerLayer(nn.Module):
    """Conformer layer adapted for MEG data."""
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        
        # Depthwise separable convolution for efficiency
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU()
        )
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, D)
        # Convolution module
        res = x
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv).transpose(1, 2)  # (B, T, D)
        x = self.ln1(x_conv + res)
        
        # Self-attention module
        res = x
        attn_out, _ = self.attention(x, x, x)
        x = self.ln2(self.dropout(attn_out) + res)
        
        # Feed-forward module
        res = x
        x = self.ffn(x)
        x = self.ln3(x + res)
        
        return x

def integrate_with_libribrain(train_dataset, val_dataset, use_zipf=True):
    """
    Example integration with LibriBrain competition code including Zipf weighting.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        use_zipf: Whether to enable Zipf-based boosting
    """
    from torch.utils.data import DataLoader
    import lightning as L
    
    # Dataset info
    dataset_info = {
        'meg_channels': 306,
        'time_points': 125,  # 0.5 seconds at 250Hz
        'num_phonemes': 39
    }
    
    # Create model with Zipf weighting
    model = create_ph_ctc_model(dataset_info, use_zipf=use_zipf)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create trainer with callbacks for monitoring
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_f1_macro',
            mode='max',
            save_top_k=3,
            filename='meg-lcs-ctc-zipf-{epoch:02d}-{val_f1_macro:.3f}'
        ),
        EarlyStopping(
            monitor='val_f1_macro',
            mode='max',
            patience=5
        )
    ]
    
    trainer = L.Trainer(
        devices="auto",
        max_epochs=20,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size of 32
        callbacks=callbacks,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model
