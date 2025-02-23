import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

print(dst_values.shape, dst_values)

# Define the sliding window data preparation, One of Time Series Cross-Validation (TSCV)
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, model_type):
        self.data = data
        self.window_size = window_size
        self.model_type = model_type

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size]  # assuming DST is the first column
        if self.model_type == "CNN1D":
            return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def sliding_window_split(data, train_window_size, val_window_size):
    total_size = len(data)
    splits = []
    for start in range(0, total_size - train_window_size - val_window_size, val_window_size):
        train_indices = np.arange(start, start + train_window_size)
        val_indices = np.arange(start + train_window_size, start + train_window_size + val_window_size)
        splits.append((train_indices, val_indices))
    return splits

def create_dataloader_from_indices(data, indices, window_size, batch_size=64, model_type="LSTM"):
    subset_data = data[indices]
    dataset = TimeSeriesDataset(subset_data, window_size, model_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Most Efficient Model until now
class RECENT_Model(nn.Module):
    def __init__(self):
        super(RECENT_Model, self).__init__()
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
        
# # LSTM Model Definition
# class LSTM_Model(nn.Module):
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
#         super(LSTM_Model, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         last_hidden_state = lstm_out[:, -1, :]
#         out = self.dropout(last_hidden_state)
#         out = self.fc(out)
#         return out

# # BiLSTM Model Definition
# class BiLSTM_Model(nn.Module):
#     def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=2, dropout=0.2):
#         super(BiLSTM_Model, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_size * 2, output_size)  # Double hidden size for bidirectional
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         last_hidden_state = lstm_out[:, -1, :]
#         out = self.dropout(last_hidden_state)
#         out = self.fc(out)
#         return out

# # GRU Model Definition
# class GRU_Model(nn.Module):
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
#         super(GRU_Model, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         gru_out, _ = self.gru(x)
#         last_hidden_state = gru_out[:, -1, :]
#         out = self.dropout(last_hidden_state)
#         out = self.fc(out)
#         return out

# LSTM+GRU Model Definition
class LSTM_GRU_Model(torch.nn.Module):
    def __init__(self, dropout_rate=0.1, hidden_size=256, num_layers=2):
        super(LSTM_GRU_Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.gru = torch.nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = torch.nn.Linear(hidden_size * 2, 1)


    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        combined = torch.cat((lstm_out, gru_out), dim=1)
        return self.fc(combined)


# Optimized 1D CNN Model with Fewer Parameters
class CNN1D_Model(nn.Module):
    def __init__(self, input_size=1, output_size=1, window_size=480):
        super(CNN1D_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Linear(64, output_size)  # Drastically reduced parameters
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  
        x = self.pool(torch.relu(self.conv2(x)))  
        x = self.pool(torch.relu(self.conv3(x)))  
        x = self.global_avg_pool(x)  # Global Average Pooling reduces feature map to size [batch, channels, 1]
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.dropout(x)
        x = self.fc(x)  
        return x  

# LSTMAttentionLSTM Model Definition
class LSTMAttentionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, attention_size=64, num_layers=2, dropout=0.2):
        super(LSTMAttentionLSTM, self).__init__()
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, dropout=dropout)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer to predict the output
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Encoder LSTM (Processing input data)
        encoder_output, (h_n, c_n) = self.encoder_lstm(x)
        
        # Attention Mechanism
        # We need to reshape the output to fit MultiheadAttention format (sequence_length, batch_size, features)
        attn_output, attn_weights = self.attention(encoder_output, encoder_output, encoder_output)
        
        # Decoder LSTM (Processing attention weighted output)
        decoder_output, _ = self.decoder_lstm(attn_output)
        
        # Fully connected layer for prediction
        out = self.fc(decoder_output[:, -1, :])  # Use the output of the last time step
        
        return out


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.early_stop_count = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
        return self.early_stop_count >= self.patience

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, model_type, fold, window_size, lr, batch_size, best_metric_value, checkpoint_dir="checkpoints", is_early_stopped=0):
    model_checkpoint_dir = os.path.join(checkpoint_dir, model_type)
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    checkpoint_file = f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_checkpoint.pth"
    if is_early_stopped:
        checkpoint_file = f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_completed.pth"
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_metric_value': best_metric_value,
    }, checkpoint_file)
    
    print(f"Checkpoint for {model_type} saved at fold {fold + 1}, epoch {epoch}")

def load_checkpoint(model, optimizer, model_type, fold, window_size, lr, batch_size, checkpoint_dir="checkpoints"):
    checkpoint_file = f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_checkpoint.pth"

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        best_metric_value = checkpoint['best_metric_value']
        
        print(f"Checkpoint for {model_type} loaded at fold {fold + 1}. Resuming from epoch {epoch}. Current Best Metric Value: {best_metric_value}")
        return model, optimizer, epoch, train_loss, val_loss, best_metric_value
    else:
        print(f"No checkpoint found for {model_type} at fold {fold + 1}. Starting from scratch.")
        return model, optimizer, 0, None, None, float('inf')

def train_with_sliding_window(data, window_size, train_window_size, val_window_size, lr=1e-3, batch_size=64, num_epochs=100, model_type="LSTM"):
    splits = sliding_window_split(data, train_window_size, val_window_size)

    val_losses = []
    mae_scores = []
    best_model = None
    best_model_info = {}
    best_metric_value = float('inf')
    best_model_info["best_val_loss"] = best_metric_value

    all_models_info = []

    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"Training fold {fold + 1}/{len(splits)}...")

        writer = SummaryWriter(log_dir=f"runs/{model_type}/window{window_size}_lr{lr}_batch{batch_size}_fold{fold}")

        train_loader = create_dataloader_from_indices(data, train_indices, window_size, batch_size, model_type)
        val_loader = create_dataloader_from_indices(data, val_indices, window_size, batch_size, model_type)

        if model_type == "LSTM":
            model = LSTM_Model().to(device)
        elif model_type == "BiLSTM":
            model = BiLSTM_Model().to(device)
        elif model_type == "GRU":
            model = GRU_Model().to(device)
        elif model_type == "RECENT":
            model = RECENT_Model().to(device)
        elif model_type == "CNN1D":
            model = CNN1D_Model(window_size=window_size).to(device)
        elif model_type == "LSTM_GRU":
            model = LSTM_GRU_Model().to(device)
        elif model_type == "LSTM_ATTENTION_LSTM":
            model = LSTMAttentionLSTM().to(device)
        else:
            return print("Unknown model")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=5, delta=0.001)

        if os.path.exists(f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_completed.pth"):
            print("That Model is completed, so go into next step...")
            continue

        start_epoch = 0
        model, optimizer, start_epoch, train_loss, _, best_metric_value = load_checkpoint(model, optimizer, model_type, fold, window_size, lr, batch_size)

        val_loss = None
        mae = None
        
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}...")
            model.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(x_batch)
                output = output.reshape(-1)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if len(train_loader) > 0:
                train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            all_preds = []
            all_true = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    output = model(x_batch)
                    output = output.reshape(-1)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
                    all_preds.append(output.cpu().numpy())
                    all_true.append(y_batch.cpu().numpy())

            if len(val_loader) > 0:
                val_loss /= len(val_loader)

            if len(all_preds) > 0 and len(all_true) > 0:
                all_preds = np.concatenate(all_preds, axis=0)
                all_true = np.concatenate(all_true, axis=0)
                mae = mean_absolute_error(all_true, all_preds)
            else:
                mae = np.nan

            writer.add_scalar('Train Loss', train_loss, epoch + 1)
            writer.add_scalar('Validation Loss', val_loss, epoch + 1)
            writer.add_scalar('MAE', mae, epoch + 1)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, MAE: {mae:.4f}")

            if early_stopping(val_loss):
                print("Early stopping triggered!")
                save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, model_type, fold, window_size, lr, batch_size, best_metric_value, is_early_stopped=1)
                break
            
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, model_type, fold, window_size, lr, batch_size, best_metric_value)

            if val_loss < best_metric_value:
                best_metric_value = val_loss
                best_model = model.state_dict()

                best_model_info = {
                    "model_type": model_type,
                    "window_size": window_size,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "fold": fold,
                    "best_val_loss": best_metric_value,
                    "mae": mae,
                    "epochs_trained": epoch + 1,
                    "model_weights": f"best_model_{model_type}_window{window_size}_lr{lr}_batch{batch_size}_fold{fold}.pth"
                }
        
        if val_loss is not None and mae is not None:
            model_info = {
                "model_type": model_type,
                "window_size": window_size,
                "learning_rate": lr,
                "batch_size": batch_size,
                "fold": fold,
                "val_loss": val_loss,
                "mae": mae
            }
            all_models_info.append(model_info)
            val_losses.append(val_loss)
            mae_scores.append(mae)

    avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else np.nan
    avg_mae = np.mean(mae_scores) if len(mae_scores) > 0 else np.nan
    print(f"Average Validation Loss: {avg_val_loss:.4f}, Average MAE: {avg_mae:.4f}")

    writer.add_scalar('Average Validation Loss', avg_val_loss, num_epochs + 1)
    writer.add_scalar('Average MAE', avg_mae, num_epochs + 1)

    if best_model is not None:
        print(f"Saving the best model with validation loss: {best_metric_value:.4f}")
        torch.save(best_model, f"best_model.pth")

        best_model_info_serializable = {
            key: float(value) if isinstance(value, np.float32) else value
            for key, value in best_model_info.items()
        }

        # Save the best model information in a JSON file
        with open("best_model.json", 'w') as f:
            json.dump(best_model_info_serializable, f, indent=4)
        
        print(f"Best model information saved to best_model.json")
    
    writer.close()

# number of fold
num_fold = 3
val_window_size = len(dst_values) // (8 + 2 * num_fold) * 2  # Size of validation window
train_window_size = val_window_size * 4  # Size of training window, train: 80%, val: 20% of each fold

# Hyperparameter tuning
learning_rates = {
    "RECENT": [1e-4, 1e-3],
    "LSTM_GRU": [1e-3, 1e-4],
    "CNN1D": [1e-4, 3e-3, 1e-3],
    "LSTM_ATTENTION_LSTM": [1e-4, 3e-3, 1e-3],    
}

window_sizes = {
    "RECENT": [48, 168],
    "LSTM_GRU": [480, 168],
    "CNN1D": [480, 168, 48],
    "LSTM_ATTENTION_LSTM": [480, 168, 48],    
}

batch_sizes = {
    "RECENT": [512, 64],
    "LSTM_GRU": [512, 64],
    "CNN1D": [512, 64],
    "LSTM_ATTENTION_LSTM": [512, 64],    
}
# Model List
model_list = ["RECENT", "LSTM_GRU", "CNN1D", "LSTM_ATTENTION_LSTM"]

for model_type in model_list:
    for lr in learning_rates[model_type]:
        for window_size in window_sizes[model_type]:
            for batch_size in batch_sizes[model_type]:
                print(f"Training with window_size={window_size}, lr={lr}, batch_size={batch_size}")
                train_with_sliding_window(dst_values, window_size, train_window_size, val_window_size, lr=lr, batch_size=batch_size, model_type=model_type)