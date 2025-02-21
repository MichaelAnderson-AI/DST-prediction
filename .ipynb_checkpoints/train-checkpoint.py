import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Import models
from models.LSTM_model import *
from models.BiLSTM_model import *
from models.CNN1D_model import *
from models.ARIMA_model import *


# Load DST Index Data
def load_dst_data(file_path):
    data = pd.read_csv(file_path)
    data["ds"] = pd.to_datetime(data["ds"])
    data.set_index("ds", inplace=True)
    return data

# Load DST dataset (Fine-tuning data)
dst_data = load_dst_data("dst_data_1975_2025.csv")

# Prepare Data for LSTM Model
dst_values = dst_data["y"].values

# Step 1: Normalization (using MinMaxScaler to scale between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dst_values_scaled = scaler.fit_transform(dst_values.reshape(-1, 1))  # Reshaping for the scaler

# Step 2: Prepare the sliding window for LSTM input
window_size = 64
X, y = [], []

# Create sequences of 64 values (inputs) and the next value (output)
for i in range(window_size, len(dst_values_scaled)):
    X.append(dst_values_scaled[i - window_size:i, 0])  # Get 64 previous values
    y.append(dst_values_scaled[i, 0])  # Get the current value (to predict)

X = np.array(X)
y = np.array(y)

# Step 3: Reshape X to be compatible with LSTM input format (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (num_samples, 64, 1)

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Step 6: Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Now the data is ready for training!

# Initialize the model
model = LSTM_MODEL(input_size=1, hidden_size=12, num_layers=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Batch training 
    train_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs = inputs.unsqueeze(2)  # Add channel dimension for LSTM (batch, seq_len, 1)
        
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets.unsqueeze(1))  # targets should also have the shape (batch_size, 1)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if batch_idx % 1000 == 0:
            print(f"batch_idx: {batch_idx}, loss: {loss}")

    # Compute average loss for the epoch
    train_loss /= len(train_loader)
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")

    # Validation
    # model.eval()
    # with torch.no_grad():
    #     val_output = model(val)
