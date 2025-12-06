"""
Simplified Enhanced UniPhyNet with all metrics as scalars for WandB
All metrics will show up as regular charts in WandB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy, ConfusionMatrix
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.last_attention = None

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).squeeze(-1)).unsqueeze(-1)
        out = avg_out + max_out
        attention = self.sigmoid(out)
        self.last_attention = attention.detach()
        return attention * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_attention = None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        attention = self.sigmoid(out)
        self.last_attention = attention.detach()
        return attention * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out
    
    def get_attention_stats(self):
        """Get attention statistics as scalars"""
        stats = {}
        if self.ca.last_attention is not None:
            stats['channel_attention_mean'] = self.ca.last_attention.mean().item()
            stats['channel_attention_std'] = self.ca.last_attention.std().item()
        if self.sa.last_attention is not None:
            stats['spatial_attention_mean'] = self.sa.last_attention.mean().item()
            stats['spatial_attention_std'] = self.sa.last_attention.std().item()
        return stats


class ResNet_1D_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 downsampling, reduction_ratio=16, block_id=0):
        super(ResNet_1D_CBAM_Block, self).__init__()
        self.block_id = block_id
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.downsampling = downsampling
        
        if downsampling:
            self.downsample_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        else:
            self.downsample_layer = None
            
        self.cbam = CBAM(out_channels, reduction_ratio)
        
        # For gradient tracking
        self.register_buffer('grad_norm', torch.tensor(0.0))

    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        if out.size(2) > 2:
            out = F.max_pool1d(out, kernel_size=2, stride=2)
        
        out = self.cbam(out)
        
        if self.downsampling and self.downsample_layer is not None:
            identity = self.downsample_layer(identity)
            if identity.size(2) > out.size(2):
                identity = F.adaptive_avg_pool1d(identity, out.size(2))
        elif not self.downsampling:
            if identity.size(2) > out.size(2):
                identity = F.adaptive_avg_pool1d(identity, out.size(2))
            
        if identity.shape != out.shape:
            if identity.size(1) != out.size(1):
                identity = F.pad(identity, (0, 0, 0, out.size(1) - identity.size(1)))
            if identity.size(2) != out.size(2):
                identity = F.adaptive_avg_pool1d(identity, out.size(2))
                
        out += identity
        return out


class UniPhyNetMEGBackbone(nn.Module):
    def __init__(
        self, 
        kernels=[3, 5, 7],
        samples=125,
        num_feature_maps=128,
        res_blocks=4,
        in_channels=306,
        fixed_kernel_size=5,
        num_classes=39,
        dropout_rate=0.5,
        use_rnn=True,
        rnn_hidden_size=128,
        rnn_layers=2
    ):
        super(UniPhyNetMEGBackbone, self).__init__()
        self.kernels = kernels
        self.planes = num_feature_maps
        self.in_channels = in_channels
        self.use_rnn = use_rnn
        self.num_classes = num_classes
        
        # For feature tracking
        self.feature_stats = {}
        
        self.parallel_conv = nn.ModuleList()
        for kernel_size in kernels:
            conv = nn.Conv1d(in_channels, self.planes, kernel_size, 
                           stride=1, padding=kernel_size//2, bias=False)
            self.parallel_conv.append(conv)
            
        self.bn1 = nn.BatchNorm1d(self.planes * len(kernels))
        self.relu = nn.SiLU(inplace=False)
        
        self.conv1 = nn.Conv1d(self.planes * len(kernels), self.planes, 
                               fixed_kernel_size, stride=1, padding=fixed_kernel_size//2, bias=False)
        
        # Create ResNet blocks with IDs for tracking
        self.blocks = nn.ModuleList()
        for i in range(res_blocks):
            downsample = (i == 0)
            self.blocks.append(ResNet_1D_CBAM_Block(
                self.planes, self.planes, fixed_kernel_size, 1, 
                fixed_kernel_size // 2, downsampling=downsample, block_id=i
            ))
        
        self.bn2 = nn.BatchNorm1d(self.planes)
        self.avgpool = nn.AdaptiveAvgPool1d(16)
        
        flattened_size = self.planes * 16
        
        if use_rnn:
            self.rnn = nn.GRU(self.in_channels, hidden_size=rnn_hidden_size, 
                             num_layers=rnn_layers, bidirectional=True, 
                             dropout=0.3 if rnn_layers > 1 else 0)
            combined_size = flattened_size + 2 * rnn_hidden_size
            self.rnn_features_size = 2 * rnn_hidden_size
        else:
            self.rnn = None
            combined_size = flattened_size
            self.rnn_features_size = 0
        
        self.cnn_features_size = flattened_size
        self.fc1 = nn.Linear(combined_size, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, track_features=False):
        original_x = x
        
        # Multi-scale convolutions with tracking
        kernel_outputs = []
        for i, conv in enumerate(self.parallel_conv):
            conv_out = conv(x)
            if conv_out.shape[2] != x.shape[2]:
                conv_out = F.interpolate(conv_out, size=x.shape[2], mode='linear', align_corners=False)
            kernel_outputs.append(conv_out)
            
            if track_features:
                self.feature_stats[f'kernel_{self.kernels[i]}_mean'] = conv_out.mean().item()
                self.feature_stats[f'kernel_{self.kernels[i]}_std'] = conv_out.std().item()
            
        out = torch.cat(kernel_outputs, dim=1)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        
        if track_features:
            self.feature_stats['after_conv1_mean'] = out.mean().item()
            self.feature_stats['after_conv1_std'] = out.std().item()
        
        # ResNet blocks with attention tracking
        for i, block in enumerate(self.blocks):
            out = block(out)
            if track_features:
                attention_stats = block.cbam.get_attention_stats()
                for key, val in attention_stats.items():
                    self.feature_stats[f'block_{i}_{key}'] = val
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        cnn_features = out.reshape(out.shape[0], -1)
        
        if track_features:
            self.feature_stats['cnn_features_mean'] = cnn_features.mean().item()
            self.feature_stats['cnn_features_std'] = cnn_features.std().item()
        
        # Store CNN-only features for component analysis
        self.last_cnn_features = cnn_features.detach()
        
        # RNN processing
        if self.use_rnn:
            rnn_out, _ = self.rnn(original_x.permute(0, 2, 1))
            rnn_features = rnn_out[:, -1, :]
            
            if track_features:
                self.feature_stats['rnn_features_mean'] = rnn_features.mean().item()
                self.feature_stats['rnn_features_std'] = rnn_features.std().item()
            
            self.last_rnn_features = rnn_features.detach()
            combined = torch.cat([cnn_features, rnn_features], dim=1)
        else:
            self.last_rnn_features = None
            combined = cnn_features
        
        # Classification
        fc1_out = self.fc1(combined)
        fc1_out = self.relu(fc1_out)
        
        if track_features:
            self.feature_stats['fc1_activation_mean'] = fc1_out.mean().item()
            self.feature_stats['fc1_activation_sparsity'] = (fc1_out == 0).float().mean().item()
        
        result = self.dropout(fc1_out)
        result = self.fc2(result)
        
        return result


class UniPhyNetMEG(L.LightningModule):
    """
    Enhanced UniPhyNetMEG with comprehensive scalar metrics for WandB
    """
    def __init__(
        self,
        time_points=125,
        learning_rate=1e-4,
        weight_decay=1e-5,
        num_classes=39,
        channels=306,
        kernels=[3, 5, 7],
        num_feature_maps=128,
        res_blocks=4,
        fixed_kernel_size=5,
        dropout_rate=0.5,
        use_rnn=True,
        rnn_hidden_size=128,
        rnn_layers=2,
        label_smoothing=0.0,
        use_scheduler='onecycle',
        scheduler_params=None,
        optimizer='adamw',
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        log_interval=50,  # How often to log detailed metrics
        track_components=True,  # Track CNN vs RNN contribution
        track_phonemes=True,  # Track per-phoneme performance
        track_gradients=True,  # Track gradient flow
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = UniPhyNetMEGBackbone(
            kernels=kernels,
            samples=time_points,
            num_feature_maps=num_feature_maps,
            res_blocks=res_blocks,
            in_channels=channels,
            fixed_kernel_size=fixed_kernel_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_rnn=use_rnn,
            rnn_hidden_size=rnn_hidden_size,
            rnn_layers=rnn_layers
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Standard metrics
        self.train_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.train_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = Accuracy(num_classes=num_classes, task='multiclass')
        
        # Per-class metrics
        self.train_f1_per_class = F1Score(num_classes=num_classes, average='none', task='multiclass')
        self.val_f1_per_class = F1Score(num_classes=num_classes, average='none', task='multiclass')
        
        # Confusion matrix
        self.val_confusion = ConfusionMatrix(num_classes=num_classes, task='multiclass')
        
        # Top-k accuracy
        self.val_top3_acc = Accuracy(num_classes=num_classes, task='multiclass', top_k=3)
        self.val_top5_acc = Accuracy(num_classes=num_classes, task='multiclass', top_k=5)
        
        # For tracking
        self.best_val_f1 = 0.0
        self.log_interval = log_interval
        
        # Phoneme performance tracking
        self.phoneme_correct = torch.zeros(num_classes)
        self.phoneme_total = torch.zeros(num_classes)
        self.phoneme_confidence = defaultdict(list)
        self.confusion_counts = torch.zeros(num_classes, num_classes)
        
        # Component contribution tracking
        self.cnn_correct = 0
        self.rnn_correct = 0
        self.combined_correct = 0
        self.component_samples = 0
        
        # Gradient tracking
        self.gradient_norms = {}
        if track_gradients:
            self.register_gradient_hooks()
        
        # Training dynamics
        self.batch_count = 0
        self.epoch_count = 0
    
    def register_gradient_hooks(self):
        """Register hooks to track gradient norms"""
        def make_hook(name):
            def hook(grad):
                if grad is not None:
                    self.gradient_norms[name] = grad.norm().item()
            return hook
        
        # Track key layer gradients
        self.model.fc2.weight.register_hook(make_hook('fc2'))
        self.model.fc1.weight.register_hook(make_hook('fc1'))
        self.model.conv1.weight.register_hook(make_hook('conv1'))
        for i, conv in enumerate(self.model.parallel_conv):
            conv.weight.register_hook(make_hook(f'kernel_{self.model.kernels[i]}'))
        for i, block in enumerate(self.model.blocks):
            block.conv1.weight.register_hook(make_hook(f'block_{i}'))
    
    def forward(self, x, track_features=False):
        return self.model(x, track_features=track_features)
    
    def compute_component_contributions(self, x, y):
        """Compute CNN-only, RNN-only, and combined accuracy"""
        if not self.hparams.use_rnn or not self.hparams.track_components:
            return {}
        
        with torch.no_grad():
            # Get full prediction
            full_output = self(x)
            combined_pred = full_output.argmax(dim=1)
            combined_correct = (combined_pred == y).float().mean()
            
            # CNN-only prediction (zero out RNN features)
            if self.model.last_cnn_features is not None:
                zeros = torch.zeros(x.size(0), self.model.rnn_features_size, device=x.device)
                cnn_only_combined = torch.cat([self.model.last_cnn_features, zeros], dim=1)
                cnn_only_fc1 = self.model.relu(self.model.fc1(cnn_only_combined))
                cnn_only_output = self.model.fc2(cnn_only_fc1)
                cnn_only_pred = cnn_only_output.argmax(dim=1)
                cnn_only_correct = (cnn_only_pred == y).float().mean()
                
                # RNN-only prediction (zero out CNN features)
                zeros = torch.zeros(x.size(0), self.model.cnn_features_size, device=x.device)
                rnn_only_combined = torch.cat([zeros, self.model.last_rnn_features], dim=1)
                rnn_only_fc1 = self.model.relu(self.model.fc1(rnn_only_combined))
                rnn_only_output = self.model.fc2(rnn_only_fc1)
                rnn_only_pred = rnn_only_output.argmax(dim=1)
                rnn_only_correct = (rnn_only_pred == y).float().mean()
                
                return {
                    'cnn_only_acc': cnn_only_correct.item(),
                    'rnn_only_acc': rnn_only_correct.item(),
                    'combined_acc': combined_correct.item(),
                    'cnn_vs_combined_ratio': cnn_only_correct.item() / (combined_correct.item() + 1e-6),
                    'rnn_vs_combined_ratio': rnn_only_correct.item() / (combined_correct.item() + 1e-6),
                }
        
        return {}
    
    def update_phoneme_stats(self, y_hat, y):
        """Update per-phoneme statistics"""
        if not self.hparams.track_phonemes:
            return
        
        probs = F.softmax(y_hat, dim=-1)
        predictions = y_hat.argmax(dim=-1)
        
        # Update per-phoneme accuracy
        for i in range(len(y)):
            true_label = y[i].item()
            pred_label = predictions[i].item()
            confidence = probs[i, pred_label].item()
            
            self.phoneme_total[true_label] += 1
            if pred_label == true_label:
                self.phoneme_correct[true_label] += 1
            
            self.phoneme_confidence[true_label].append(confidence)
            self.confusion_counts[true_label, pred_label] += 1
    
    def log_phoneme_metrics(self, phase='train'):
        """Log phoneme-level metrics as scalars"""
        if not self.hparams.track_phonemes:
            return
        
        metrics = {}
        
        # Per-phoneme accuracy
        phoneme_accs = []
        for i in range(self.hparams.num_classes):
            if self.phoneme_total[i] > 0:
                acc = (self.phoneme_correct[i] / self.phoneme_total[i]).item()
                phoneme_accs.append(acc)
                # Log top 5 worst and best phonemes
                metrics[f'{phase}/phoneme_{i}_acc'] = acc
        
        if phoneme_accs:
            sorted_accs = sorted(phoneme_accs)
            metrics[f'{phase}/worst_phoneme_acc'] = sorted_accs[0]
            metrics[f'{phase}/bottom_5_phoneme_acc'] = np.mean(sorted_accs[:5]) if len(sorted_accs) >= 5 else sorted_accs[0]
            metrics[f'{phase}/bottom_10_phoneme_acc'] = np.mean(sorted_accs[:10]) if len(sorted_accs) >= 10 else np.mean(sorted_accs)
            metrics[f'{phase}/median_phoneme_acc'] = np.median(sorted_accs)
            metrics[f'{phase}/top_10_phoneme_acc'] = np.mean(sorted_accs[-10:]) if len(sorted_accs) >= 10 else np.mean(sorted_accs)
            metrics[f'{phase}/best_phoneme_acc'] = sorted_accs[-1]
            metrics[f'{phase}/phoneme_acc_std'] = np.std(phoneme_accs)
            
        # Top confusion pairs
        if self.confusion_counts.sum() > 0:
            # Mask diagonal (correct predictions)
            confusion_masked = self.confusion_counts.clone()
            confusion_masked[torch.eye(self.hparams.num_classes, dtype=torch.bool)] = 0
            
            # Find top confusions
            top_k = 5
            flat_confusion = confusion_masked.flatten()
            top_confusion_values, top_confusion_indices = torch.topk(flat_confusion, top_k)
            
            for i, (val, idx) in enumerate(zip(top_confusion_values, top_confusion_indices)):
                if val > 0:
                    true_idx = idx // self.hparams.num_classes
                    pred_idx = idx % self.hparams.num_classes
                    metrics[f'{phase}/top_confusion_{i+1}_count'] = val.item()
                    metrics[f'{phase}/top_confusion_{i+1}_pair'] = true_idx.item() * 100 + pred_idx.item()  # Encode as single number
        
        # Confidence statistics
        all_confidences = []
        for conf_list in self.phoneme_confidence.values():
            all_confidences.extend(conf_list)
        
        if all_confidences:
            metrics[f'{phase}/mean_confidence_all'] = np.mean(all_confidences)
            metrics[f'{phase}/std_confidence_all'] = np.std(all_confidences)
            metrics[f'{phase}/min_confidence'] = np.min(all_confidences)
            metrics[f'{phase}/max_confidence'] = np.max(all_confidences)
        
        self.log_dict(metrics, on_step=False, on_epoch=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        self.batch_count += 1
        
        # Track features periodically
        track_features = (batch_idx % self.log_interval == 0)
        
        y_hat = self(x, track_features=track_features)
        loss = self.criterion(y_hat, y)
        
        # Standard metrics
        f1 = self.train_f1(y_hat, y)
        acc = self.train_acc(y_hat, y)
        
        # Update phoneme stats
        self.update_phoneme_stats(y_hat, y)
        
        # Detailed analysis every N batches
        if batch_idx % self.log_interval == 0:
            # Prediction quality metrics
            probs = F.softmax(y_hat, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            
            # Top-k predictions
            top_k_values, top_k_indices = torch.topk(probs, k=min(5, self.hparams.num_classes), dim=-1)
            top1_in_top5 = (y.unsqueeze(1) == top_k_indices).any(dim=1).float().mean()
            
            # Log immediate metrics
            self.log_dict({
                'train/mean_confidence': probs.max(dim=-1)[0].mean(),
                'train/prediction_entropy': entropy,
                'train/top1_in_top5': top1_in_top5,
                'train/batch_loss': loss,  # Current batch loss
            }, on_step=True, on_epoch=False)
            
            # Feature statistics
            if track_features and self.model.feature_stats:
                feature_metrics = {f'features/{k}': v for k, v in self.model.feature_stats.items()}
                self.log_dict(feature_metrics, on_step=True, on_epoch=False)
            
            # Gradient norms
            if self.gradient_norms:
                grad_metrics = {f'gradients/{k}': v for k, v in self.gradient_norms.items()}
                self.log_dict(grad_metrics, on_step=True, on_epoch=False)
                
                # Log gradient health metrics
                grad_values = list(self.gradient_norms.values())
                if grad_values:
                    self.log_dict({
                        'gradients/mean_norm': np.mean(grad_values),
                        'gradients/max_norm': np.max(grad_values),
                        'gradients/min_norm': np.min(grad_values),
                        'gradients/norm_ratio': np.max(grad_values) / (np.min(grad_values) + 1e-6),
                    }, on_step=True, on_epoch=False)
            
            # Component contribution analysis (less frequent)
            if self.hparams.use_rnn and batch_idx % (self.log_interval * 5) == 0:
                # Use small batch for efficiency
                subset_size = min(8, x.size(0))
                component_stats = self.compute_component_contributions(x[:subset_size], y[:subset_size])
                if component_stats:
                    comp_metrics = {f'components/{k}': v for k, v in component_stats.items()}
                    self.log_dict(comp_metrics, on_step=True, on_epoch=False)
        
        # Standard logging
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1_macro', f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Standard metrics
        f1 = self.val_f1(y_hat, y)
        acc = self.val_acc(y_hat, y)
        
        # Per-class F1
        f1_per_class = self.val_f1_per_class(y_hat, y)
        
        # Top-k accuracy
        top3_acc = self.val_top3_acc(y_hat, y)
        top5_acc = self.val_top5_acc(y_hat, y)
        
        # Update confusion matrix
        self.val_confusion.update(y_hat, y)
        
        # Update phoneme stats
        self.update_phoneme_stats(y_hat, y)
        
        # Prediction quality
        probs = F.softmax(y_hat, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        self.log('val_top3_acc', top3_acc)
        self.log('val_top5_acc', top5_acc)
        self.log('val/mean_confidence', probs.max(dim=-1)[0].mean())
        self.log('val/prediction_entropy', entropy)
        
        # Track best F1
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            self.log('best_val_f1', self.best_val_f1)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log detailed metrics at epoch end"""
        self.epoch_count += 1
        
        # Log phoneme metrics
        self.log_phoneme_metrics('train')
        
        # Log epoch summary
        self.log_dict({
            'epoch': float(self.epoch_count),
            'total_batches_seen': float(self.batch_count),
        }, on_step=False, on_epoch=True)
        
        # Reset phoneme stats for next epoch
        self.phoneme_correct.zero_()
        self.phoneme_total.zero_()
        self.phoneme_confidence.clear()
        self.confusion_counts.zero_()
        
        # Reset metrics
        self.train_f1.reset()
        self.train_acc.reset()
        self.train_f1_per_class.reset()
    
    def on_validation_epoch_end(self):
        """Log validation metrics"""
        # Get per-class F1 scores
        f1_scores = self.val_f1_per_class.compute()
        
        # Log F1 distribution statistics
        self.log_dict({
            'val/f1_min': f1_scores.min(),
            'val/f1_max': f1_scores.max(),
            'val/f1_std': f1_scores.std(),
            'val/f1_bottom_10_mean': f1_scores.topk(10, largest=False)[0].mean(),
            'val/f1_top_10_mean': f1_scores.topk(10, largest=True)[0].mean(),
        }, on_step=False, on_epoch=True)
        
        # Log confusion matrix statistics
        conf_matrix = self.val_confusion.compute()
        
        # Diagonal accuracy (per-class accuracy from confusion matrix)
        diagonal = conf_matrix.diag()
        row_sums = conf_matrix.sum(dim=1)
        per_class_acc = diagonal / (row_sums + 1e-6)
        
        self.log_dict({
            'val/confusion_diagonal_mean': per_class_acc.mean(),
            'val/confusion_diagonal_std': per_class_acc.std(),
            'val/confusion_off_diagonal_sum': (conf_matrix.sum() - diagonal.sum()) / (conf_matrix.sum() + 1e-6),
        }, on_step=False, on_epoch=True)
        
        # Log phoneme metrics
        self.log_phoneme_metrics('val')
        
        # Reset metrics
        self.val_f1.reset()
        self.val_acc.reset()
        self.val_f1_per_class.reset()
        self.val_confusion.reset()
        self.val_top3_acc.reset()
        self.val_top5_acc.reset()
        
        # Reset phoneme stats
        self.phoneme_correct.zero_()
        self.phoneme_total.zero_()
        self.phoneme_confidence.clear()
        self.confusion_counts.zero_()
    
    def configure_optimizers(self):
        # Separate parameter groups
        no_decay = ['bias', 'bn', 'norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n.lower() for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n.lower() for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # Select optimizer
        if self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate
            )
        
        # Configure scheduler
        if self.hparams.use_scheduler == 'onecycle':
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'estimated_stepping_batches'):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 1000 * 50
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        return optimizer