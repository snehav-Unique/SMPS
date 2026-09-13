import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet implementation in PyTorch.
    Original paper: EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces
    """
    def __init__(self, num_classes=4, channels=46, samples=2500, F1=8, D=2, F2=16, kernel_length=64):
        super(EEGNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Depthwise Conv
        self.depthwise = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=0.25)
        
        # Separable Conv (implemented as a grouped conv followed by 1x1 conv)
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 16 // 2), groups=F1 * D, bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=0.25)
        
        # Calculate fully connected layer input size
        # samples = 2500 -> pool1 (1, 4) = 625 -> pool2 (1, 8) = 78
        out_samples = (samples // 4) // 8
        self.fc = nn.Linear(F2 * out_samples, num_classes)
        
    def forward(self, x):
        # Input shape expected: (Batch, 1, Channels, Time)
        
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
