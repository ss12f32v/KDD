import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaEncoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        """Define layers for a vanilla rnn encoder"""
        super(VanillaEncoder, self).__init__()

        self.gru = nn.GRU(input_size= 6, 
                          hidden_size= hidden_size)

    def forward(self, input_seqs, hidden=None):
        outputs, hidden = self.gru(input_seqs, hidden)

        return outputs, hidden

   