from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
logging.getLogger("prophet.plot").disabled = True
from fiber.logging_utils import get_logger
logger = get_logger(__name__)
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import pickle
from huggingface_hub import HfApi, hf_hub_download
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

class CustomGeomagneticModel:
    def __init__(self):
        """Initialize your custom geomagnetic model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        model_path = hf_hub_download(repo_id="smartguy0505/dst-index-predictor", filename="dst_lstm_model_update.pth")

        self.model = BidirectionalLSTM().to(self.device)

        if not torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()  # Set the model to evaluation mode


    def run_inference(self, dst_data: pd.DataFrame) -> Dict[str, Any]:
        dst_data["y"] = dst_data["y"] * 100

        dst_values = dst_data["y"].values
        # Normalize Data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dst_values.reshape(-1, 1))

        future_input = torch.tensor(scaled_data[-24:].reshape(1, 24, 1), dtype=torch.float32).to(self.device)
        future_prediction = self.model(future_input)

        future_prediction = scaler.inverse_transform(future_prediction.detach().cpu().numpy().reshape(-1, 1))

        predicted_value = future_prediction[0][0]

        print(f"Predicted Value: {predicted_value}")
        return {
            "predicted_value": float(predicted_value) / 100
        }


if __name__ == "__main__":
    new_data_file = 'datasets/test_new.csv'  # Replace with your CSV file path
    new_df = pd.read_csv(new_data_file, index_col='ds', parse_dates=True)
    model = CustomGeomagneticModel()
    model.run_inference(new_df)