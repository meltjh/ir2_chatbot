import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


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
    sorted_input, mapping2 = self.sort_inputs(embedded_input)
    packed_input = pack_sequence(sorted_input)

    # Model-model
    lstm_output, (lstm_hidden, _) = self.lstm(packed_input)

    # Post-model
    hidden_bilinear_input, hidden_bilinear_templates, mapping1 = self.unpack_bilinear(lstm_hidden, mapping1, mapping2, num_inputs, templates)
    hidden_concats = self.unpack_decoder(lstm_output, mapping1, mapping2, num_inputs)

    return hidden_bilinear_input, hidden_bilinear_templates, hidden_concats, mapping1

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

    return hidden_bilinear_input, hidden_bilinear_templates, mapping1

  def unpack_decoder(self, lstm_output, mapping1, mapping2, num_inputs):
    unpacked_output, lengths = pad_packed_sequence(lstm_output)

    hidden_concat = []
    for i in range(len(lengths)):
      hidden_concat.append(unpacked_output[:lengths[i],i].contiguous().view(lengths[i]*self.hidden_size*2))

    hidden_concat_ordered = np.array(hidden_concat)[mapping2]
    # hidden_concat_ordered = np.array(hidden_concat)[mapping2]
    # hidden_concat_input = hidden_concat_ordered[:num_inputs]
    #
    # hidden_concat_templates = []
    # for input_templates in mapping1:
    #   hidden_concat_templates.append(hidden_concat_ordered[input_templates])

    return hidden_concat_ordered


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

class BahdanauAttnDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
    super(BahdanauAttnDecoderRNN, self).__init__()
    #TODO max_length definieren
    max_length = 10

    # Define parameters
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.max_length = max_length

    # Define layers
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.dropout = nn.Dropout(dropout_p)
    self.attn =  nn.Linear(self.hidden_size * 2, self.max_length)
    self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, word_input, last_hidden, encoder_outputs):
    # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs

    # Get the embedding of the current input word (last output word)
    word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
    word_embedded = self.dropout(word_embedded)

    # Calculate attention weights and apply to encoder outputs
    attn_weights = self.attn(last_hidden[-1], encoder_outputs)
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

    # Combine embedded input word and attended context, run through RNN
    rnn_input = torch.cat((word_embedded, context), 2)
    output, hidden = self.gru(rnn_input, last_hidden)

    # Final output layer
    output = output.squeeze(0) # B x N
    output = F.log_softmax(self.out(torch.cat((output, context), 1)))

    # Return final output, hidden state, and attention weights (for visualization)
    return output, hidden, attn_weights

class Model(nn.Module):
  def __init__(self, word2id, hidden_dim=500, embedding_dim=500):
    super().__init__()
    self.word2id = word2id
    self.encoder = Encoder(len(self.word2id.id2w), embedding_dim, hidden_dim)
    self.bilinear = Bilinear(hidden_dim)

  def forward(self, input, templates):
    batch_size = len(input)
    hidden_bilinear_input, hidden_bilinear_templates, hidden_concats, mapping1 = self.encoder(input, templates)
    output_states, saliency = self.bilinear(hidden_bilinear_input, hidden_bilinear_templates, mapping1)

    return saliency
