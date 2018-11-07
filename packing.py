import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_sequence

embs = nn.Embedding(10, 3)
rnn = nn.RNN(3, 100, num_layers=2, batch_first=True)

inputs = [[1], [2, 5, 9, 2], [1, 2], [9, 9, 9]]
templates = [[[2, 5, 9, 2], [1, 2], [9, 9, 9]], [[2, 5, 9, 2], [1, 2], [9, 9, 9]]]

inputs = [embs(torch.tensor(i)) for i in inputs]
templates = list(map(lambda l: [embs(torch.tensor(t)) for t in l], templates))

def pack(sequences):
  order = list(reversed(sorted(range(len(sequences)), key=lambda i: len(sequences[i]))))
  sorted_sequences = np.array(sequences)[order]
  packed_sequences = pack_sequence(sorted_sequences)

  # Determine reverse index to restore original order
  reverse_order = torch.empty(len(order)).long()
  for i, j in enumerate(order):
    reverse_order[j] = i

  return packed_sequences, reverse_order

packed_inputs, order = pack(inputs)
h, i = rnn(packed_inputs)
print(order)
print(i[1,:])
print(i[1, order])


# sentences = reversed(sorted(sentences, key=len))
# emb_sentences = [embs(torch.tensor(s)) for s in sentences]
# packed_sequences = pack(emb_sentences)
