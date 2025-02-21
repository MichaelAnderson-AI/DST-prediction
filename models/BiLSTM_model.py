import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
        
        # BiLSTM layer (bidirectional=True)
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, 1)  # Since it's bidirectional, we multiply by 2
        
    def forward(self, x):
        # BiLSTM expects input of shape (batch_size, seq_length, input_size)
        bilstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Use the last hidden state (or the output of the last LSTM cell)
        output = self.fc(bilstm_out[:, -1, :])
        
        return output

# Initialize the BiLSTM model
bi_lstm_model = BiLSTMModel(input_size=1, hidden_size=64, num_layers=2)
