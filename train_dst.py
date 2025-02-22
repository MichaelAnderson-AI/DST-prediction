import os
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
# Sliding Window Cross-Validation (for more accurate measurement of one model)
# NOTE: you can use other options: Expanding Window Cross-Validation, Blocked Cross-Validation, ... for TSCV
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
        # return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
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
        
# LSTM Model Definition
import torch
import torch.nn as nn

# LSTM Model Definition
class LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.dropout(last_hidden_state)
        out = self.fc(out)
        return out


# BiLSTM Model Definition
class BiLSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=2, dropout=0.2):
        super(BiLSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Double hidden size for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.dropout(last_hidden_state)
        out = self.fc(out)
        
        return out

# GRU Model Definition
class GRU_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden_state = gru_out[:, -1, :]
        out = self.dropout(last_hidden_state)
        out = self.fc(out)
        return out



# 1D CNN Model Definition
class CNN1D_Model(nn.Module):
    def __init__(self, input_size=1, output_size=1, window_size=24):
        super(CNN1D_Model, self).__init__()
        
        # 1D Convolution layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * (window_size // 8), 128)  # Adjusting size after pooling
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Apply conv2 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv3(x)))  # Apply conv3 + ReLU + Pooling
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc2(x)  # Final output layer
        return x


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
    # Create model-specific directory for checkpoints if not exists & Save the checkpoint along with best_metric_value.
    model_checkpoint_dir = os.path.join(checkpoint_dir, model_type)
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    # Format the checkpoint filename using the hyperparameters
    checkpoint_file = f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_checkpoint.pth"
    if is_early_stopped:
        checkpoint_file = f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_completed.pth"
        
    # Save the checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_metric_value': best_metric_value,  # Save the best metric value
    }, checkpoint_file)
    
    print(f"Checkpoint for {model_type} saved at fold {fold + 1}, epoch {epoch}")


def load_checkpoint(model, optimizer, model_type, fold, window_size, lr, batch_size, checkpoint_dir="checkpoints"):

    # Define checkpoint filename with formatted hyperparameters
    checkpoint_file = f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_checkpoint.pth"

    # Check if checkpoint exists
    if os.path.exists(checkpoint_file):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        best_metric_value = checkpoint['best_metric_value']  # Load best_metric_value
        
        print(f"Checkpoint for {model_type} loaded at fold {fold + 1}. Resuming from epoch {epoch}. Current Best Metric Value: {best_metric_value}")
        return model, optimizer, epoch, train_loss, val_loss, best_metric_value
    else:
        print(f"No checkpoint found for {model_type} at fold {fold + 1}. Starting from scratch.")
        return model, optimizer, 0, None, None, float('inf') # Start with inf for best_metric_value


import json
def train_with_sliding_window(data, window_size, train_window_size, val_window_size, lr=1e-3, batch_size=64, num_epochs=100, model_type="LSTM"):
    splits = sliding_window_split(data, train_window_size, val_window_size)

    val_losses = []
    mae_scores = []
    best_model = None
    best_model_info = {}  # Store information about the best model
    best_metric_value = float('inf')  # Initialize with infinity for best model selection
    best_model_info["best_val_loss"] = best_metric_value

    # Store information about all models for comparison (not saving them yet)
    all_models_info = []

    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"Training fold {fold + 1}/{len(splits)}...")

        # TensorBoard writer - Separate log directories for each fold and model type
        writer = SummaryWriter(log_dir=f"runs/{model_type}/window{window_size}_lr{lr}_batch{batch_size}_fold{fold}")

        train_loader = create_dataloader_from_indices(data, train_indices, window_size, batch_size, model_type)
        val_loader = create_dataloader_from_indices(data, val_indices, window_size, batch_size, model_type)

        # Model selection
        if model_type == "LSTM":
            model = LSTM_Model().to(device)  # 1 input feature for DST values
        elif model_type == "BiLSTM":
            model = BiLSTM_Model().to(device)  # 1 input feature for DST values
        elif model_type == "GRU":
            model = GRU_Model().to(device)  # 1 input feature for DST values
        elif model_type == "RECENT":
            model = RECENT_Model().to(device)  # 1 input feature for DST values        
        elif model_type == "CNN1D":
            model = CNN1D_Model(window_size=window_size).to(device)  # 1 input feature for DST values        
        else:
            return print("Unknown model")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=5, delta=0.001)

        if os.path.exists(f"checkpoints/{model_type}/window{window_size}_lr{lr:.5f}_batch{batch_size}_fold{fold}_completed.pth"):
            print("That Model is completed, so go into next step...")
            continue
        # Checkpoint filename for each fold and model
        start_epoch = 0
        model, optimizer, start_epoch, train_loss, _, best_metric_value = load_checkpoint(model, optimizer, model_type, fold, window_size, lr, batch_size)

        # Initialize variables for val_loss and mae before training
        val_loss = None
        mae = None
        
        # Training loop
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

            # Avoid division by zero if no training steps are completed
            if len(train_loader) > 0:
                train_loss /= len(train_loader)

            # Validation loop
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

            # Avoid division by zero if no validation steps are completed
            if len(val_loader) > 0:
                val_loss /= len(val_loader)

            # MAE Calculation - Handle case if no predictions are made
            if len(all_preds) > 0 and len(all_true) > 0:
                all_preds = np.concatenate(all_preds, axis=0)
                all_true = np.concatenate(all_true, axis=0)
                mae = mean_absolute_error(all_true, all_preds)
            else:
                mae = np.nan  # Handle case where no validation predictions are made

            # TensorBoard Logging
            writer.add_scalar('Train Loss', train_loss, epoch + 1)
            writer.add_scalar('Validation Loss', val_loss, epoch + 1)
            writer.add_scalar('MAE', mae, epoch + 1)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, MAE: {mae:.4f}")

            # Early stopping
            if early_stopping(val_loss):
                print("Early stopping triggered!")
                save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, model_type, fold, window_size, lr, batch_size, best_metric_value, is_early_stopped=1)
                break
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, model_type, fold, window_size, lr, batch_size, best_metric_value)

            # Track the best model based on validation loss or MAE
            if val_loss < best_metric_value:  # Use avg_val_loss as metric
                best_metric_value = val_loss  # Choose based on validation loss
                best_model = model.state_dict()  # Save the best model weights

                # Save the best model information (hyperparameters, validation loss, MAE, etc.)
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
        
        # Save the comprehensive model info for each fold for later comparison
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

    # Calculate and log the average validation loss and MAE across all folds
    avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else np.nan
    avg_mae = np.mean(mae_scores) if len(mae_scores) > 0 else np.nan
    print(f"Average Validation Loss: {avg_val_loss:.4f}, Average MAE: {avg_mae:.4f}")

    # Log average values to TensorBoard
    writer.add_scalar('Average Validation Loss', avg_val_loss, num_epochs + 1)
    writer.add_scalar('Average MAE', avg_mae, num_epochs + 1)

    # Save only the best model's information and weights
    if best_model is not None:
        print(f"Saving the best model with validation loss: {best_metric_value:.4f}")
        torch.save(best_model, f"best_model.pth")

        # Save the best model information in a JSON file
        with open("best_model.json", 'w') as f:
            json.dump(best_model_info, f, indent=4)
        
        print(f"Best model information saved to best_model.json")
    
    writer.close()  # Close TensorBoard writer


# number of fold
num_fold = 3
val_window_size = len(dst_values) // (8 + 2 * num_fold) * 2  # Size of validation window
train_window_size = val_window_size * 4  # Size of training window, train: 80%, val: 20% of each fold

# Hyperparameter tuning
window_sizes = [24, 48, 96]
learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [512, 64, 8]
# Model List
model_list = ["LSTM", "BiLSTM", "GRU", "CNN1D", "RECENT"]

for model_type in model_list:
        for window_size in window_sizes:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    print(f"Training with window_size={window_size}, lr={lr}, batch_size={batch_size}")
                    train_with_sliding_window(dst_values, window_size, train_window_size, val_window_size, lr=lr, batch_size=batch_size, model_type=model_type)


