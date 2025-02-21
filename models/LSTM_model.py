import torch.nn as nn
import torch.optim as optim

class LSTM_MODEL(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTM_MODEL, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)  # Single output (predicted Dst)
        
    def forward(self, x):
        # LSTM expects input of shape (batch_size, seq_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state to make prediction (or the output of the last LSTM cell)
        output = self.fc(lstm_out[:, -1, :])
        
        return output

# Initialize the model
# model = LSTM_model(input_size=1, hidden_size=64, num_layers=2)

