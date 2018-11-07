import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class Encoder(nn.Module):
  def __init__(self, embeddings, hidden_size=500):
    super().__init__()

    self.hidden_size = hidden_size
    self.emb = embeddings
    self.lstm = nn.LSTM(self.emb.embedding_dim, self.hidden_size, num_layers=2, bidirectional=True)

  def forward(self, input):
    embedded_input = [self.emb(x) for x in input]
    packed_sequences, restore_order = self.pack(embedded_input)
    lstm_output, (lstm_hidden, _) = self.lstm(packed_sequences)

    # For bilinear
    final_hidden = lstm_hidden.view(2, 2, -1, self.hidden_size)[1].view(-1, self.hidden_size*2)

    # For decoder
    hidden_concat = self.unpack(lstm_output)

    return final_hidden, hidden_concat

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

  def unpack(self, packed_sequence):
    unpacked_sequence, lengths = pad_packed_sequence(packed_sequence)

    hidden_concats = []
    for i in range(len(lengths)):
      hidden_concats.append(unpacked_sequence[:lengths[i],i].contiguous().view(lengths[i]*self.hidden_size*2))

    return hidden_concats

class Bilinear(nn.Module):
  def __init__(self, hidden_size=500):
    super().__init__()

    self.bifc = nn.Bilinear(2*hidden_size, 2*hidden_size, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x1, x2):
    bi_out = self.bifc(x1, x2)

    return self.sigmoid(bi_out)

x = [[4, 5, 2], [2, 1], [1, 1, 1, 1]]
x = [torch.tensor(i) for i in x]
emd = nn.Embedding(10, 500)
e = Encoder(emd)
b = Bilinear()
u, j = e(x)
s = b(u, u)

print(j)
for t in j:
  print(t.size())
