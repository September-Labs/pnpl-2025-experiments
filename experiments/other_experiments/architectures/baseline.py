import torch
import torch.nn as nn
import lightning as L
from torchmetrics import F1Score


class PhonemeClassificationModel(L.LightningModule):
    """
    Baseline Lightning model for phoneme classification from MEG data.
    
    Architecture:
    - Conv1d: 306 channels -> 128 channels (1x1 convolution)
    - ReLU activation
    - Flatten: (128, time_points) -> (128 * time_points,)
    - Linear: (128 * time_points) -> 39 phoneme classes
    
    Input: (batch_size, 306, time_points) - 306 MEG channels, variable time points
    Output: (batch_size, 39) - logits for 39 phoneme classes
    """
    
    def __init__(self, time_points=None, learning_rate=0.0005):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model layers that don't depend on time_points
        self.conv1 = nn.Conv1d(306, 128, 1)  # 1x1 convolution
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Initialize linear layer as None - will be created on first forward pass
        self.linear = None
        self.time_points = time_points
        
        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes=39, average='macro', task="multiclass")
        
    def forward(self, x):
        # Get the time dimension from input
        batch_size, channels, time_dim = x.shape
        
        # Create linear layer on first forward pass if needed
        if self.linear is None:
            # Calculate the flattened size after conv and flatten
            with torch.no_grad():
                dummy_input = torch.zeros(1, channels, time_dim)
                dummy_output = self.flatten(self.relu(self.conv1(dummy_input)))
                flattened_size = dummy_output.shape[1]
            
            self.linear = nn.Linear(flattened_size, 39).to(x.device)
            print(f"Created linear layer with input size {flattened_size} (time_points={time_dim})")
        
        # Forward pass
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1_macro)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)