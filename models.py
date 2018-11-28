import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from utils import sort_by_length

class Encoder(nn.Module):
  def __init__(self, embeddings: nn.Embedding, hidden_size: int):
    super().__init__()

    self.hidden_size = hidden_size
    self.embeddings = embeddings

    self.lstm = nn.LSTM(self.embeddings.embedding_dim, self.hidden_size, num_layers=2, bidirectional=True)

  def forward(self, input: list, templates: list):
    # Stack all inputs and templates into one list to process them as batch
    stacked_input, unstack_mapping = self.stack_inputs(input, templates)

    # Embed the input and template words
    embedded_input = [self.embeddings(x) for x in stacked_input]

    # Sort the stacked inputs and templates by length, allowing them to be packed
    sorted_input, unsort_mapping, _ = sort_by_length(embedded_input)

    # Pack the inputs to allow the variable length sequences to be processed as a batch
    packed_input = pack_sequence(sorted_input)

    # Encode the inputs and templates with the LSTM
    _, (lstm_hidden, _) = self.lstm(packed_input)

    # Reshape the output to (num_layers, num_directions, batch_size, hidden_size) and take
    # and concat the final state in each direction of layer 2
    lstm_hidden = lstm_hidden.view(2, 2, -1, self.hidden_size)
    sorted_encodings = lstm_hidden[1].view(-1, self.hidden_size * 2)

    # Unsort the output to its original stacked order
    encodings = sorted_encodings[unsort_mapping]

    # Match the input and template encodings into two tensors for the bilinear module
    input_encodings, template_encodings = self.create_bilinear_tensors(encodings, unstack_mapping, templates)

    return input_encodings, template_encodings, unstack_mapping

  def create_bilinear_tensors(self, encodings, unstack_mapping, templates):
    # Repeat the indices of the input for the number of templates it has
    input_indices = []
    for i in range(len(templates)):
      input_indices += [i]*len(templates[i])
    template_indices = [item for sublist in unstack_mapping for item in sublist]

    # Index from encodings
    input_encodings = encodings[input_indices]
    template_encodings = encodings[template_indices]

    return input_encodings, template_encodings

  @staticmethod
  def stack_inputs(input, templates):
    # Place input at top of list
    stacked = [] + input
    mapping = []

    # Place templates for input below
    index = len(stacked)
    for batch_templates in templates:
      stacked += batch_templates
      mapping.append(list(range(index, index+len(batch_templates))))

      index += len(batch_templates)

    return stacked, mapping

class Bilinear(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()

    self.bifc = nn.Bilinear(2*hidden_size, 2*hidden_size, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x_input, x_templates, mapping1):
    batch_size = len(mapping1)
    saliency = self.sigmoid(self.bifc(x_input, x_templates))

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
  def __init__(self, word2id, hidden_size, embedding_size, embeddings_matrix):
    super().__init__()
    self.word2id = word2id
    self.vocab_size = len(self.word2id.id2w)

    self.embeddings = nn.Embedding(self.vocab_size, embedding_size)
    self.embeddings.weight.data.copy_(torch.from_numpy(embeddings_matrix))
    self.embeddings.requires_grad = False
    self.encoder = Encoder(self.embeddings, hidden_size)
    self.bilinear = Bilinear(hidden_size)
    self.decoder = Decoder(self.embeddings, hidden_size)

  def forward(self, input, target, templates):
    bilinear_inputs, bilinear_templates, stack_mapping = self.encoder(input, templates)
    template_state, saliency = self.bilinear(bilinear_inputs, bilinear_templates, stack_mapping)
    response = self.decoder(target, template_state)

    return saliency, response

  def respond(self, device, word2id, input, templates, max_length=100):
    bilinear_input, bilinear_templates, stack_mapping = self.encoder(input, templates)
    template_state, saliency = self.bilinear(bilinear_input, bilinear_templates, stack_mapping)

    return self.decoder.generate(word2id, template_state, max_length, device)


class Decoder(nn.Module):
  def __init__(self, embeddings, hidden_size):
    super().__init__()

    self.embeddings = embeddings
    self.gru = nn.GRU(self.embeddings.embedding_dim, 4*hidden_size)
    self.out = nn.Linear(4*hidden_size, self.embeddings.num_embeddings)
    self.softmax = nn.LogSoftmax(-1)

  def forward(self, target, template_state):
    embedded_targets = [self.embeddings(x[:-1]) for x in target]
    sorted_targets, sort_map_backward, sort_map_forward = sort_by_length(embedded_targets)
    sorted_template_states = template_state[sort_map_forward]
    packed_targets = pack_sequence(sorted_targets)
    gru_output, _ = self.gru(packed_targets, sorted_template_states.unsqueeze(0))
    unpacked_gru_output, sequence_lengths = self.unpack_gru_output(gru_output)
    softmax_output = self.softmax(self.out(unpacked_gru_output))
    split_output = self.split_softmax_output(softmax_output, sequence_lengths)
    output = split_output[sort_map_backward]

    return output

  def generate(self, word2id, hidden_state, max_length, device):
    hidden_state = hidden_state.unsqueeze(0)
    sentence = [torch.tensor([word2id.tag_id_bos]).long().to(device)]

    for i in range(max_length-2):
      prev_word = self.embeddings(sentence[-1]).unsqueeze(0)
      gru_output, hidden_state = self.gru(prev_word, hidden_state)
      linear_out = self.out(gru_output)
      softmax_output = self.softmax(linear_out)
      next_word = softmax_output.argmax().unsqueeze(0)
      sentence.append(next_word)

      if next_word == word2id.tag_id_eos:
        break

    if len(sentence) > max_length-2:
      sentence.append(torch.tensor([word2id.tag_id_eos]).long().to(device))

    sentence = torch.cat(sentence)
    string = word2id.id2string(sentence)

    return sentence, string

  @staticmethod
  def unpack_gru_output(gru_output):
    gru_padded_output, sequence_lengths = pad_packed_sequence(gru_output)

    batch_indices = []
    word_indices = []
    for batch, sequence_length in enumerate(sequence_lengths):
      batch_indices += [batch]*sequence_length.item()
      word_indices += list(range(sequence_length))

    stacked_unpadded_output = gru_padded_output[word_indices, batch_indices]

    return stacked_unpadded_output, sequence_lengths

  @staticmethod
  def split_softmax_output(softmax_output, sequence_lengths):
    split_output = []
    start_index = 0
    for sequence_length in sequence_lengths:
      split_output.append(softmax_output[start_index:start_index+sequence_length])
      start_index += sequence_length

    return np.array(split_output)
