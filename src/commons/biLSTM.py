import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, dropout=0.3, bidirectional=True):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, x):
        output, _ = self.rnn(x)
        return output
    

if __name__ == '__main__':
    x = torch.randn(8, 100, 128)

    lengths = torch.tensor([100, 95, 90, 85, 80, 75, 70, 65])
    bilstm_layer = BiLSTMLayer(input_size=128, hidden_size=512, num_layers=3)

    output = bilstm_layer(x, lengths)
    print("Output shape:", output.shape)