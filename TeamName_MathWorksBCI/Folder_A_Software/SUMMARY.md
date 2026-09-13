# Summary

## Preprocessing & Data Loading
Due to time constraints and the large size of the official IEEE dataset (~47 GB), a complete training run was not feasible prior to this deadline. However, a robust data ingestion pipeline was built. The `train_model.py` script is designed to handle the continuous 6-minute `.npz` recordings, extract the `MarkOnSignal` event codes, and dynamically slice the data into 5-second (2500 sample) trials. 

## Handling Variable Channels
The pipeline dynamically accepts both (2500, 29) and (2500, 46) array shapes. To ensure tensor shape consistency for the deep learning model, 29-channel data is automatically padded with zeros up to 46 channels before processing.

## Feature Extraction & Model Architecture
We implemented a PyTorch-based **EEGNet** architecture (`eegnet.py`). EEGNet is highly compact, utilizing Depthwise and Separable Convolutions, making it extremely efficient for Motor Imagery tasks and ideal for potential Edge Deployment (e.g., Raspberry Pi). 

## Training Strategy & Current Status
The current `predictions.csv` was generated using the automated `run_inference.py` pipeline with the instantiated EEGNet architecture prior to weight convergence (untrained weights / baseline), allowing us to validate the end-to-end inference pipeline on the test set while the 47GB training dataset finishes downloading.
