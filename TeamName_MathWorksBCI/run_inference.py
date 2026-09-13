import os
import sys
import numpy as np
import csv
import torch

# Add Folder_A_Software to path to import eegnet
sys.path.append(os.path.join(os.path.dirname(__file__), "Folder_A_Software"))
try:
    from eegnet import EEGNet
except ImportError:
    print("Warning: eegnet.py not found. Ensure it is inside Folder_A_Software/")
    EEGNet = None

def load_model(model_path, device):
    if EEGNet is None:
        return None
    model = EEGNet(num_classes=4, channels=46, samples=2500)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded trained model weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using UNTRAINED weights.")
        model.to(device)
        model.eval()
    return model

def main():
    base_dir = os.path.dirname(__file__)
    test_data_dir = os.path.join(os.path.dirname(base_dir), "test_data")
    output_csv = os.path.join(base_dir, "Folder_A_Software", "predictions.csv")
    model_path = os.path.join(base_dir, "Folder_A_Software", "models", "eegnet_model.pth")
    
    classes = ["left_hand", "right_hand", "feet", "idle"]
    
    if not os.path.exists(test_data_dir):
        print(f"Error: {test_data_dir} does not exist.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    predictions = []
    
    print(f"Starting inference on test data using device: {device}...")
    for i in range(1, 2001):
        test_id = f"test_{i:05d}"
        file_name = f"{test_id}.npz"
        file_path = os.path.join(test_data_dir, file_name)
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            signal = data["signal"] # Shape: (2500, channels)
            
            if model is not None:
                # 1. Pad 29 channels to 46 if necessary
                if signal.shape[1] == 29:
                    padded = np.zeros((2500, 46))
                    padded[:, :29] = signal
                    signal = padded
                
                # 2. Reshape for PyTorch: (Batch, 1, Channels, Time)
                signal = signal.T # Now (46, 2500)
                signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                signal_tensor = signal_tensor.to(device)
                
                # 3. Predict
                with torch.no_grad():
                    outputs = model(signal_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_label = classes[predicted.item()]
            else:
                import random
                predicted_label = random.choice(classes)
        else:
            print(f"Warning: {file_path} not found. Predicting 'idle'.")
            predicted_label = "idle"
            
        predictions.append({"test_id": test_id, "predicted_label": predicted_label})
        
        if i % 500 == 0:
            print(f"Processed {i}/2000 samples.")
            
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_id", "predicted_label"])
        writer.writeheader()
        writer.writerows(predictions)
        
    print(f"Successfully wrote predictions to {output_csv}")

if __name__ == "__main__":
    main()
