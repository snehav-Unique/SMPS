import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from eegnet import EEGNet

class BCIChallengeDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        
        self.data = []
        self.labels = []
        
        # Standard BCI event codes
        # 769: left hand, 770: right hand, 771: feet. 
        # We assume 780 or 786 is idle/rest for this dataset based on unique codes.
        # You may need to adjust the 'idle' mapping based on the dataset's official description.
        event_map = {
            769: 0, # left_hand
            770: 1, # right_hand
            771: 2, # feet
            780: 3  # idle (assuming 780 is rest/idle)
        }
        
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        print(f"Found {len(npz_files)} files in {data_dir}. Loading...")
        
        for file_path in npz_files:
            try:
                d = np.load(file_path, allow_pickle=True)
                signal = d['signal'] # (Time, Channels)
                marks = d['MarkOnSignal'] # (num_events, 2) [sample_index, event_code]
                
                for mark in marks:
                    start_idx = int(mark[0])
                    event_code = int(mark[1])
                    
                    if event_code in event_map:
                        label = event_map[event_code]
                        end_idx = start_idx + 2500 # 5 seconds at 500Hz
                        
                        if end_idx <= signal.shape[0]:
                            epoch_signal = signal[start_idx:end_idx, :]
                            ch = epoch_signal.shape[1]
                            
                            # Pad 29 channels to 46
                            if ch == 29:
                                padded = np.zeros((2500, 46))
                                padded[:, :29] = epoch_signal
                                epoch_signal = padded
                                
                            # Transpose to (Channels, Time) for PyTorch
                            epoch_signal = epoch_signal.T
                            
                            self.data.append(epoch_signal)
                            self.labels.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"Loaded {len(self.data)} total extracted trials.")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.expand_dims(self.data[idx], axis=0) 
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

def train():
    data_dir = r"d:\RVCE\Sem 3\MathWorksChallenge\MathWorks\SMPS\train_data"
    batch_size = 16
    epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    train_dataset = BCIChallengeDataset(data_dir, mode="train")
    if len(train_dataset) == 0:
        print("No training data found. Please download the dataset first.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = EEGNet(num_classes=4, channels=46, samples=2500).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    os.makedirs("models", exist_ok=True)
    save_path = "models/eegnet_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
