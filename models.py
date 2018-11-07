import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_sequence

class Encoder(nn.Module):
  def __init__(self, embeddings, hidden_size=500):
    super().__init__()

    self.hidden_size = hidden_size
    self.emb = embeddings
    self.lstm = nn.LSTM(self.emb.embedding_dim, self.hidden_size, num_layers=2, bidirectional=True)

  def forward(self, input):
    embedded_input = [self.emb(x) for x in input]
    packed_sequences, restore_order = self.pack(embedded_input)
    _, (lstm_hidden, _) = self.lstm(packed_sequences)
    final_hidden = lstm_hidden.view(2, 2, -1, self.hidden_size)[1].view(-1, self.hidden_size*2)

    return final_hidden

  @staticmethod
  def pack(input):
    order = list(reversed(sorted(range(len(input)), key=lambda i: len(input[i]))))
    sorted_sequences = np.array(input)[order]
    packed_sequences = pack_sequence(sorted_sequences)

    # Determine reverse index to restore original order
    reverse_order = torch.empty(len(order)).long()
    for i, j in enumerate(order):
      reverse_order[j] = i

    return packed_sequences, reverse_order

x = [[4, 5, 2], [2, 1], [1, 1, 1, 1]]
x = [torch.tensor(i) for i in x]
emd = nn.Embedding(10, 500)
e = Encoder(emd)
e(x)
