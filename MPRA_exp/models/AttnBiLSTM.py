import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        input = input.permute(0,2,1)
        
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        
        output = output.permute(0,2,1)
        return output


class AttnBiLSTM(nn.Module):
    def __init__(
            self, 
            input_length = 50, 
            output_dim=1, 
            motif_conv_hidden = 256, 
            conv_hidden = 128, 
            n_heads = 8, 
            conv_width_motif = 30, 
            dropout_rate = 0.2
            ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=motif_conv_hidden, kernel_size=conv_width_motif, padding='same')
        self.norm1 = nn.BatchNorm1d(motif_conv_hidden)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=motif_conv_hidden, out_channels=conv_hidden, kernel_size=conv_width_motif, padding='same')
        self.norm2 = nn.BatchNorm1d(conv_hidden)
        self.relu2 = nn.ReLU()

        # (batch_size, hidden_dim, length) --> (length, batch_size, hidden_dim)
        self.attention1 = nn.TransformerEncoderLayer(d_model=conv_hidden, nhead=n_heads, batch_first=True)
        self.attention2 = nn.TransformerEncoderLayer(d_model=conv_hidden, nhead=n_heads, batch_first=True)
        
        self.bilstm = BiLSTM(input_size=conv_hidden, hidden_size=conv_hidden, output_size=int(conv_hidden//4)) # torch.Size([10, 32, 110])
        self.flatten1 = nn.Flatten() # torch.Size([10, 3520])
        self.dense1 = nn.Linear(int(conv_hidden//4) * input_length, conv_hidden)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear( conv_hidden, conv_hidden)
        self.relu4 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.dense3 = nn.Linear(conv_hidden, output_dim)
        
        
    def forward(self,x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        
        x = x.permute(2,0,1)
        x = self.attention1(x)
        x = self.attention2(x)
        x = x.permute(1,2,0)
        
        x = self.bilstm(x)
        x = self.relu3(self.dense1(self.flatten1(x)))
        x = self.relu4(self.dense2(self.drop1(x)))
        x = self.dense3(self.drop2(x))
        # x = x.squeeze(-1)
        return x