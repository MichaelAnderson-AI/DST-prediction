import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Progress bar

# Enable GPU Acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Load DST Index Data
def load_dst_data(file_path):
    data = pd.read_csv(file_path)
    data["ds"] = pd.to_datetime(data["ds"])
    data.set_index("ds", inplace=True)
    return data

# Load DST dataset
dst_data = load_dst_data("data_new.csv")

# Prepare Data for LSTM Model
dst_values = dst_data["y"].values

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dst_values.reshape(-1, 1))

# Create Dataset (Using last 60 time steps to predict the next one)
X_data, y_data = [], []
for i in range(60, len(scaled_data)):  
    X_data.append(scaled_data[i-60:i, 0])  # Past 60 values as input
    y_data.append(scaled_data[i, 0])  # Predict next time step

X_data, y_data = np.array(X_data), np.array(y_data)

# Reshape X_data correctly to (num_samples, 60, 1)
X_data = X_data.reshape(X_data.shape[0], 60, 1)

# Convert to PyTorch Tensors
X_data = torch.tensor(X_data, dtype=torch.float32)
y_data = torch.tensor(y_data, dtype=torch.float32)

# Create a TensorDataset and DataLoader for batch processing
batch_size = 64
train_dataset = TensorDataset(X_data, y_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the Model (Bidirectional LSTM)
class BidirectionalLSTM(nn.Module):
    def __init__(self):
        super(BidirectionalLSTM, self).__init__()
        self.lstm1 = nn.LSTM(1, 512, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(512 * 2, 256, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(256 * 2, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.fc1(x[:, -1, :])  # Get the last hidden state
        x = self.fc2(x)
        return x

# Load Existing Model or Train New One
model_path = "dst_lstm_model.pth"
model = BidirectionalLSTM().to(device)

if os.path.exists(model_path):
    print("Loading existing model...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Fine-tuning with new data...")
else:
    print("Training a new model...")

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
epochs = 10  
model.train()

for epoch in range(epochs):
    total_loss = 0  
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")  # Progress bar

    for X_batch, y_batch in progress_bar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  # Show loss in progress bar

    print(f"Epoch [{epoch+1}/{epochs}] Completed - Average Loss: {total_loss/len(train_loader):.6f}")

    # Save the model at each epoch
    torch.save(model.state_dict(), model_path)
    print(f"Model checkpoint saved at epoch {epoch+1}.")

# Final model save
torch.save(model.state_dict(), model_path)
print("Final model saved after training.")

# Predict Next Value (using the last 60 time steps)
model.eval()
future_input = torch.tensor(scaled_data[-60:].reshape(1, 60, 1), dtype=torch.float32).to(device)
future_prediction = model(future_input)

# Convert prediction back to original scale
future_prediction = scaler.inverse_transform(future_prediction.detach().cpu().numpy().reshape(-1, 1))

# Display the prediction
print(f"\nPredicted DST Index for next time step: {future_prediction[0][0]:.3f}")

# Save future prediction
future_df = pd.DataFrame({"Time": [dst_data.index[-1] + pd.Timedelta(hours=1)], "Predicted DST Index": future_prediction.flatten()})
future_df.to_csv("dst_future_prediction.csv", index=False)
print("Future prediction saved to dst_future_prediction.csv")

# Plot Historical Data and Future Prediction
plt.figure(figsize=(16, 8))
plt.title("DST Index Prediction")
plt.xlabel("Time", fontsize=14)
plt.ylabel("DST Index", fontsize=14)
plt.plot(dst_data.index, dst_data["y"], label="Historical DST Index")
plt.plot([dst_data.index[-1] + pd.Timedelta(hours=1)], future_prediction, label="Next Time Step Prediction", marker='o', linestyle='dashed', color='red')
plt.legend()
plt.show()
