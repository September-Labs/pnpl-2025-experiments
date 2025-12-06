import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score


class SimpleCNN(nn.Module):
    """Simple CNN model for MEG phoneme classification (similar to tutorial)"""
    
    def __init__(self, num_channels=306, time_points=125, num_classes=39, dropout=0.2):
        super().__init__()
        
        # 1D convolution along time dimension
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
        self.flatten_size = 128 * time_points  # 128 * 125 = 16000
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        x = self.conv1(x)  # (batch, 128, time)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.flatten(1)  # (batch, 128 * time)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class PhonemeClassificationSimpleCNN(L.LightningModule):
    """Lightning module wrapper for Simple CNN"""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        # Model configuration
        model_params = config.get('simple_cnn_config', {})
        self.model = SimpleCNN(
            num_channels=306,
            time_points=125,  # 0.5s at 250Hz
            num_classes=39,
            dropout=model_params.get('dropout', 0.2)
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_f1 = F1Score(task="multiclass", num_classes=39, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=39, average='macro')
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.0005)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        f1 = self.train_f1(preds, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        f1 = self.val_f1(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        f1 = self.val_f1(preds, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)