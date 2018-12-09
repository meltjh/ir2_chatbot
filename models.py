import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from utils import sort_by_length

class Encoder(nn.Module):
  def __init__(self, embeddings: nn.Embedding, hidden_size: int, use_bilinear: bool):
    super().__init__()

    self.use_bilinear = use_bilinear
    
    self.hidden_size = hidden_size
    self.embeddings = embeddings

    self.lstm = nn.LSTM(self.embeddings.embedding_dim, self.hidden_size, num_layers=2, bidirectional=True)

  def forward(self, input: list, templates: list):
    
    if self.use_bilinear:
      # Stack all inputs and templates into one list to process them as batch
      stacked_input, unstack_mapping = self.stack_inputs(input, templates)
    else:
      unstack_mapping = None
      stacked_input = input
    
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

    
    if self.use_bilinear:
      # Match the input and template encodings into two tensors for the bilinear module
      input_encodings, template_encodings, bilinear_mapping = self.create_bilinear_tensors(encodings, unstack_mapping, templates)
      return input_encodings, template_encodings, bilinear_mapping
    else:
      return encodings, None, None
    
    return input_encodings, template_encodings, bilinear_mapping

  def create_bilinear_tensors(self, encodings, unstack_mapping, templates):
    batch_size = len(templates)

    # Repeat the indices of the input for the number of templates it has
    input_indices = []
    for i in range(batch_size):
      input_indices += [i]*len(templates[i])
    template_indices = [item for sublist in unstack_mapping for item in sublist]

    # Index from encodings
    input_encodings = encodings[input_indices]
    template_encodings = encodings[template_indices]

    # Create new index for bilinear mapping
    bilinear_mapping = [[i-batch_size for i in batch_ids] for batch_ids in unstack_mapping]

    return input_encodings, template_encodings, bilinear_mapping

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

  def forward(self, input_encodings, template_encodings, bilinear_mapping):
    # Run encodings through bilinear layer to obtain predicted saliency between input and templates
    saliency = self.sigmoid(self.bifc(input_encodings, template_encodings))

    # The unstack mapping is for the tensor with the inputs at the top. In this module the encodings have been
    batched_weights = [F.softmax(saliency[indices], -1) for indices in bilinear_mapping]
    batched_saliency = [saliency[indices] for indices in bilinear_mapping]

    output_states = []
    for template_ids, weights in zip(bilinear_mapping, batched_weights):
      input_state = input_encodings[template_ids[0]]
      template_state = (template_encodings[template_ids] * weights).sum(0)
      output_states.append(torch.cat((input_state, template_state)))

    output_states = torch.stack(output_states)

    return output_states, batched_saliency

class Model(nn.Module):
  def __init__(self, word2id, hidden_size, embedding_size, embeddings_matrix, use_bilinear):
    super().__init__()
    self.word2id = word2id
    self.vocab_size = len(self.word2id.id2w)
    self.use_bilinear = use_bilinear

    self.embeddings = nn.Embedding(self.vocab_size, embedding_size)
    self.embeddings.weight.data.copy_(torch.from_numpy(embeddings_matrix))
    self.embeddings.requires_grad = False
    self.encoder = Encoder(self.embeddings, hidden_size, self.use_bilinear)
    if self.use_bilinear:
      self.bilinear = Bilinear(hidden_size)
    self.decoder = Decoder(self.embeddings, hidden_size, self.use_bilinear)

  def forward(self, input, target, templates):
    input_encodings, template_encodings, bilinear_mapping = self.encoder(input, templates)
    if self.use_bilinear:
      template_state, saliency = self.bilinear(input_encodings, template_encodings, bilinear_mapping)
      response = self.decoder(target, template_state)
      return saliency, response
    else:
      response = self.decoder(target, None)
      return None, response

  def respond(self, device, word2id, input, templates, max_length=100):
    bilinear_input, bilinear_templates, stack_mapping = self.encoder(input, templates)
    
    if self.use_bilinear:
      template_state, saliency = self.bilinear(bilinear_input, bilinear_templates, stack_mapping)
      return self.decoder.generate(word2id, template_state, max_length, device)
    else:
      # TODO: Hier is bilinear_input nog niet goed.
      return self.decoder.generate(word2id, bilinear_input, max_length, device)


class Decoder(nn.Module):
  def __init__(self, embeddings, hidden_size, use_bilinear):
    super().__init__()

    self.embeddings = embeddings
    self.use_bilinear = use_bilinear
    
    if self.use_bilinear:
      hidden_size *= 4
    else:
      hidden_size *= 2
    
    self.gru = nn.GRU(self.embeddings.embedding_dim, hidden_size)
    self.out = nn.Linear(hidden_size, self.embeddings.num_embeddings)
    
    self.softmax = nn.LogSoftmax(-1)

  def forward(self, target, template_state):
    embedded_targets = [self.embeddings(x[:-1]) for x in target]
    sorted_targets, sort_map_backward, sort_map_forward = sort_by_length(embedded_targets)
    packed_targets = pack_sequence(sorted_targets)
    
    if self.use_bilinear:
      sorted_template_states = template_state[sort_map_forward]
      gru_output, _ = self.gru(packed_targets, sorted_template_states.unsqueeze(0))
    else:
      gru_output, _ = self.gru(packed_targets)
       
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
      
      # hidden_state.shape is op tweede dim te groot, moet 1 zijn.
      
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
