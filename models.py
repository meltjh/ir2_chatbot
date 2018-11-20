import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


def sort_inputs(input):
  order = list(reversed(sorted(range(len(input)), key=lambda i: len(input[i]))))
  sorted_input = np.array(input)[order]

  # Determine reverse index to restore original order
  mapping = torch.empty(len(order)).long()
  for i, j in enumerate(order):
    mapping[j] = i

  return sorted_input, mapping

class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size):
    super().__init__()

    self.hidden_size = hidden_size
    self.embedding_dim = embedding_dim
    self.emb = nn.Embedding(vocab_size, self.embedding_dim)
    self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=2, bidirectional=True)

  def forward(self, input, templates):
    num_inputs = len(input)

    # Pre-model
    stacked_input, mapping1 = self.stack_inputs(input, templates)
    embedded_input = [self.emb(x) for x in stacked_input]
    sorted_input, mapping2 = sort_inputs(embedded_input)
    packed_input = pack_sequence(sorted_input)

    # Model-model
    _, (lstm_hidden, _) = self.lstm(packed_input)

    # Post-model
    hidden_bilinear_input, hidden_bilinear_templates = self.unpack_bilinear(lstm_hidden, mapping1, mapping2, num_inputs, templates)

    return hidden_bilinear_input, hidden_bilinear_templates, mapping1

  @staticmethod
  def stack_inputs(input, templates):
    stacked = input
    mapping = []

    index = len(stacked)
    for batch_templates in templates:
      stacked += batch_templates
      mapping.append(list(range(index, index+len(batch_templates))))

      index += len(batch_templates)

    return stacked, mapping

  def unpack_bilinear(self, lstm_hidden, mapping1, mapping2, num_inputs, templates):
    hidden_bilinear = lstm_hidden.view(2, 2, -1, self.hidden_size)[1].view(-1, self.hidden_size*2)[mapping2] # For bilinear
    input_indices = []
    for i in range(num_inputs):
      input_indices += [i]*len(templates[i])
    template_indices = [item for sublist in mapping1 for item in sublist]
    hidden_bilinear_input = hidden_bilinear[input_indices]
    hidden_bilinear_templates = hidden_bilinear[template_indices]

    return hidden_bilinear_input, hidden_bilinear_templates

class Bilinear(nn.Module):
  def __init__(self, hidden_size=500):
    super().__init__()

    self.bifc = nn.Bilinear(2*hidden_size, 2*hidden_size, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x_input, x_templates, mapping1):
    batch_size = len(mapping1)
    saliency = self.bifc(x_input, x_templates)

    bilinear_mapping = [[i-batch_size for i in batch_ids] for batch_ids in mapping1] # Correct because inputs are not at top for bilinear
    batched_weights = [F.softmax(saliency[indices], 0) for indices in bilinear_mapping]
    batched_saliency = [saliency[indices] for indices in bilinear_mapping]

    output_states = []
    for template_ids, weights in zip(bilinear_mapping, batched_weights):
      input_state = x_input[template_ids[0]]
      template_state = (x_templates[template_ids] * weights).sum(0)
      output_states.append(torch.cat((input_state, template_state)))

    output_states = torch.stack(output_states)

    return output_states, batched_saliency
      
      
class Model(nn.Module):
  def __init__(self, word2id, hidden_dim=500, embedding_dim=500):
    super().__init__()
    self.word2id = word2id
    self.encoder = Encoder(len(self.word2id.id2w), embedding_dim, hidden_dim)
    self.bilinear = Bilinear(hidden_dim)

  def forward(self, input, templates):
    hidden_bilinear_input, hidden_bilinear_templates, mapping1 = self.encoder(input, templates)
    output_states, saliency = self.bilinear(hidden_bilinear_input, hidden_bilinear_templates, mapping1)

    return saliency, output_states

class DecoderRNN(nn.Module):
  def __init__(self, embedding_dim, hidden_size, output_size):
    super(DecoderRNN, self).__init__()
    
    self.gru = nn.GRU(embedding_dim, hidden_size, 1)
    self.out = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, prev_output, hidden):
    output, hidden = self.gru(prev_output, hidden)
    output = self.softmax(self.out(output))
    return output, hidden
  
  
  

  
#class DecoderFullSentences(nn.Module):
#  def __init__(self, vocab_size, hidden_size, output_size, device):
#    super(DecoderFullSentences, self).__init__()
#  
#    self.device = device
#    self.vocab_size = vocab_size
#    
#    
#    self.decoder = DecoderRNN(vocab_size, hidden_size, output_size)
#  
#  def forward(self, initial_token_id, output_states, target):
#    """
#    Initial_output is the initial token index.
#    """
#    batch_size = output_states.shape[0]
#    
#    tag_bos_emb = torch.zeros(1, batch_size, self.vocab_size).to(self.device)
#    tag_bos_emb[0,:,initial_token_id] = 1
#
#    sorted_target = target
#    
#    print(len(sorted_target))
#    print(sorted_target[0].shape)
#    print(sorted_target[1].shape)
#    print(sorted_target[2].shape)
##    raise NotImplementedError("~~~~~")
#    
#    packed_output_states = pack_sequence(sorted_target)
#    
#    raise NotImplementedError("~~~~~")
#    
#    output, prev_h = self.decoder(tag_bos_emb, packed_output_states)
#    
#    print("waha")
    
    
    
    
    
    
    
    