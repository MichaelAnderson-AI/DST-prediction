import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Enable GPU Acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Load Model
model_path = "dst_lstm_model_update.pth"
model = BidirectionalLSTM().to(device)

if not torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_path))

model.eval()
print("Model loaded successfully.")

# Load and Prepare Data
def load_dst_data(file_path):
    data = pd.read_csv(file_path)
    data["ds"] = pd.to_datetime(data["ds"])
    data.set_index("ds", inplace=True)
    return data

dst_data = load_dst_data("new_train_1.csv")
dst_values = dst_data["y"].values

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dst_values.reshape(-1, 1))

# Predict Next Value (using the last 60 time steps)
future_input = torch.tensor(scaled_data.reshape(1, len(scaled_data), 1), dtype=torch.float32).to(device)
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
