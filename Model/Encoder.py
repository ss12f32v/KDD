import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaEncoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        """Define layers for a vanilla rnn encoder"""
        super(VanillaEncoder, self).__init__()

        self.gru = nn.GRU(input_size= 6, 
                          hidden_size= hidden_size)
        self.hidden_transform = nn.Linear(hidden_size, output_size)
        
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden=None):
        outputs, hidden = self.gru(input_seqs, hidden)

        hidden_tansformed = self.hidden_transform(hidden.transpose(0,1)).transpose(1,0)
        
        outputs_transformed = self.out(outputs.transpose(0,1))  # S = B x O

        return outputs_transformed, hidden, hidden_tansformed

   