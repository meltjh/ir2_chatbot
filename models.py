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

  def forward(self, input, templates):
    num_inputs = len(input)

    # Pre-model
    stacked_input, mapping1 = self.stack_inputs(input, templates)
    embedded_input = [self.emb(x) for x in stacked_input]
    sorted_input, mapping2 = self.sort_inputs(embedded_input)
    packed_input = pack_sequence(sorted_input)

    # Model-model
    lstm_output, (lstm_hidden, _) = self.lstm(packed_input)

    # Post-model
    hidden_bilinear_input, hidden_bilinear_templates = self.unpack_bilinear(lstm_hidden, mapping1, mapping2, num_inputs, templates)


    print(hidden_bilinear_templates)

    asd




    # For decoder
    hidden_concat = self.unpack(lstm_output)

    return hidden_final, hidden_concat

  def stack_inputs(self, input, templates):
    stacked = input
    mapping = []

    index = len(stacked)
    for batch_templates in templates:
      stacked += batch_templates
      mapping.append(list(range(index, index+len(batch_templates))))

      index += len(batch_templates)

    return stacked, mapping

  def sort_inputs(self, input):
    order = list(reversed(sorted(range(len(input)), key=lambda i: len(input[i]))))
    sorted_input = np.array(input)[order]

    # Determine reverse index to restore original order
    mapping = torch.empty(len(order)).long()
    for i, j in enumerate(order):
      mapping[j] = i

    return sorted_input, mapping

  def unpack_bilinear(self, lstm_hidden, mapping1, mapping2, num_inputs, templates):
    hidden_bilinear = lstm_hidden.view(2, 2, -1, self.hidden_size)[1].view(-1, self.hidden_size*2)[mapping2] # For bilinear
    input_indices = []
    for i in range(num_inputs):
      input_indices += [i]*len(templates[i])
    template_indices = [item for sublist in mapping1 for item in sublist]
    hidden_bilinear_input = hidden_bilinear[input_indices]
    hidden_bilinear_templates = hidden_bilinear[template_indices]

    return hidden_bilinear_input, hidden_bilinear_templates

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

class Decoder(nn.Module):
  def __init__(self, hidden_size=500):
    super().__init__()


  def forward(self, x1, x2):
    pass

x = [[4, 5, 2], [2, 1], [1, 1, 1, 1]]
t = [[[1, 5, 2], [1, 1, 2]], [[2, 5, 2], [2, 1, 2]], [[3, 5, 2, 1], [3, 1, 2]]]
x = [torch.tensor(i) for i in x]
t = list(map(lambda l: [torch.tensor(ti) for ti in l], t))
emd = nn.Embedding(10, 500)
e = Encoder(emd)
b = Bilinear()
u, j = e(x, t)
s = b(u, u)

print(j)
for t in j:
  print(t.size())
