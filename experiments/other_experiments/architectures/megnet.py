"""
MEGNet architecture adapted for LibriBrain MEG Phoneme Classification
Lightning module compatible with config-based training system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
import numpy as np


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

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).squeeze(-1)).unsqueeze(-1)
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out


class MEGNetBackbone(nn.Module):
    """
    EEGNet adapted for MEG phoneme classification
    Input: (batch_size, 306, time_points) - 306 MEG channels
    Output: (batch_size, 39) - logits for 39 phoneme classes
    """
    def __init__(
        self, 
        num_classes=39, 
        Chans=306, 
        Samples=125, 
        dropoutRate=0.5, 
        kernLength=25,
        F1=16,
        D=2, 
        F2=32,
        norm_rate=0.25
    ):
        super(MEGNetBackbone, self).__init__()
        
        self.F2 = F2
        self.num_classes = num_classes
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1)
        
        # Depthwise convolution across channels
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Block 2
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.batchnorm3 = nn.BatchNorm2d(F2, eps=1e-5, momentum=0.1)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # Use adaptive pooling to ensure fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        
        # Classification with fixed size
        self.flatten_size = F2 * 4
        self.classify = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Apply adaptive pooling to ensure fixed size
        x = self.adaptive_pool(x)
        
        # Classification
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        
        return x


class MEGNet(L.LightningModule):
    """
    Lightning wrapper for MEGNet model
    Compatible with config-based training system
    """
    def __init__(
        self,
        time_points=125,
        learning_rate=1e-4,
        weight_decay=1e-5,
        num_classes=39,
        channels=306,
        dropout_rate=0.5,
        kern_length=25,
        F1=16,
        D=2,
        F2=32,
        norm_rate=0.25,
        label_smoothing=0.0,
        use_scheduler='onecycle',  # 'onecycle', 'cosine', 'reduce_on_plateau', 'none'
        scheduler_params=None,
        optimizer='adamw',  # 'adamw', 'adam', 'sgd'
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = MEGNetBackbone(
            num_classes=num_classes,
            Chans=channels,
            Samples=time_points,
            dropoutRate=dropout_rate,
            kernLength=kern_length,
            F1=F1,
            D=D,
            F2=F2,
            norm_rate=norm_rate
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.test_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.train_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = Accuracy(num_classes=num_classes, task='multiclass')
        
        # For tracking best metrics
        self.best_val_f1 = 0.0
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        f1 = self.train_f1(y_hat, y)
        acc = self.train_acc(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1_macro', f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        f1 = self.val_f1(y_hat, y)
        acc = self.val_acc(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        # Track best F1
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            self.log('best_val_f1', self.best_val_f1)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        f1 = self.test_f1(y_hat, y)
        acc = self.test_acc(y_hat, y)
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        # Select optimizer
        if self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Configure scheduler
        if self.hparams.use_scheduler == 'none':
            return optimizer
        
        scheduler_params = self.hparams.scheduler_params or {}
        
        if self.hparams.use_scheduler == 'onecycle':
            # OneCycleLR needs total steps
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback estimation
                total_steps = 1000 * self.hparams.get('max_epochs', 100)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps,
                pct_start=scheduler_params.get('pct_start', 0.1),
                anneal_strategy=scheduler_params.get('anneal_strategy', 'cos')
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        elif self.hparams.use_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get('T_max', 50),
                eta_min=scheduler_params.get('eta_min', 1e-6)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        
        elif self.hparams.use_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-7)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_f1_macro',
                    'interval': 'epoch'
                }
            }
        
        return optimizer
    
    def on_train_epoch_end(self):
        # Reset metrics for next epoch
        self.train_f1.reset()
        self.train_acc.reset()
    
    def on_validation_epoch_end(self):
        # Reset metrics for next epoch
        self.val_f1.reset()
        self.val_acc.reset()
    
    def on_test_epoch_end(self):
        # Reset metrics
        self.test_f1.reset()
        self.test_acc.reset()