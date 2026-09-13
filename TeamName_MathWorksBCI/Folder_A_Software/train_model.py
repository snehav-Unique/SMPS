import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from eegnet import EEGNet

class BCIChallengeDataset(Dataset):
    """
    Dataset loader for the IEEE DataPort training data.
    You will need to replace the dummy data generation below with actual
    loading logic once you extract the dataset.
    """
    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        
        # TODO: Implement actual loading from the downloaded dataset.
        # Below is placeholder logic generating random valid-shaped arrays.
        print(f"Loading {mode} dataset from {data_dir}...")
        
        # 4 classes: 0=left_hand, 1=right_hand, 2=feet, 3=idle
        self.samples = 200 # Dummy sample count
        
        # Simulating random EEG data. 
        # Channels might be 29 or 46 in the real dataset. 
        # We simulate them mixed.
        self.data = []
        self.labels = []
        for _ in range(self.samples):
            ch = np.random.choice([29, 46])
            signal = np.random.randn(2500, ch) # (Time, Channels)
            label = np.random.randint(0, 4)
            
            # PADDING TO 46 CHANNELS
            # If the signal has 29 channels, pad it to 46
            if ch == 29:
                padded_signal = np.zeros((2500, 46))
                padded_signal[:, :29] = signal
                signal = padded_signal
            
            # Transpose to (Channels, Time) for PyTorch Conv2D
            signal = signal.T 
            
            self.data.append(signal)
            self.labels.append(label)
            
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # We need to add a "channel" dimension for PyTorch Conv2D: (1, Channels, Time)
        x = np.expand_dims(self.data[idx], axis=0) 
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

def train():
    # Parameters
    data_dir = "./path_to_ieee_dataset"
    batch_size = 16
    epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_dataset = BCIChallengeDataset(data_dir, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Instantiate Model
    model = EEGNet(num_classes=4, channels=46, samples=2500).to(device)
    
    # 3. Setup Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # 5. Save Model
    os.makedirs("models", exist_ok=True)
    save_path = "models/eegnet_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
