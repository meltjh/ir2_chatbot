import os
import torch
import rougescore
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from time import time
from word2id import Word2Id
import pickle

def print_progress(prefix: str, P, epoch: int, batch_num: int, num_batches: int, saliency_loss: float, decoder_loss: float, start_time: float) -> None:
  elapsed_time = time() - start_time
  average_rate = elapsed_time / (batch_num + 1)
  remaining_time = average_rate * (num_batches - batch_num)

  print('\r{}Epoch {:03d}/{:03d} Example {:05d}/{:05d} ({:02d}:{:02d}/{:02d}:{:02d}) | Bilinear loss: {:.4f}, Decoder loss: {:.4f}'.format(
    prefix,
    epoch+1,
    P.NUM_EPOCHS,
    (batch_num+1)*P.BATCH_SIZE,
    num_batches*P.BATCH_SIZE,
    int(elapsed_time // 60),
    int(elapsed_time % 60),
    int(remaining_time // 60),
    int(remaining_time % 60),
    saliency_loss,
    decoder_loss
  ), end='')

def unpack_batch(batch: list, device: torch.device) -> tuple:
  input, target, templates, _, saliencies = tuple(batch)

  input = [torch.tensor(i).long().to(device) for i in input]
  target = [torch.tensor(i).long().to(device) for i in target]
  templates = list(map(lambda l: [torch.tensor(i).long().to(device) for i in l], templates))
  saliencies = [torch.tensor(i).float().to(device).unsqueeze(1) for i in saliencies]

  return input, target, templates, saliencies


def save_checkpoint(P, epoch: int, model: nn.Module, optimiser: Optimizer, saliency_loss: float, decoder_loss: float) -> None:
  state = {
    'epoch': epoch + 1,
    'model': model.state_dict(),
    'optimiser': optimiser.state_dict(),
    'saliency_loss': saliency_loss,
    'decoder_loss': decoder_loss,
    'alpha': P.ALPHA
  }

  filename = '{}/cp-{}.pt'.format(P.CHECKPOINT_DIR, epoch+1)
  torch.save(state, filename)

def load_checkpoint(P, model: nn.Module, optimiser: Optimizer,  device: torch.device, specific_file = None) -> int:
  epoch = 0
  if os.path.exists(P.CHECKPOINT_DIR):
    latest_checkpoint = ""
    if specific_file != None:
      latest_checkpoint = specific_file
    else:
      checkpoints = [os.path.join(P.CHECKPOINT_DIR, f) for f in os.listdir(P.CHECKPOINT_DIR) if f.endswith('.pt')]
  
      if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
      else:
        return epoch
   
    state = torch.load(latest_checkpoint, map_location=device)

    model.load_state_dict(state['model'])
    optimiser.load_state_dict(state['optimiser'])
    epoch = state['epoch']
    
    model.train()

  return epoch

def get_saliency(target, templates):
  """
  Input: The targets is a list of word-id's and templates a list of list of word-id's
  Returns the saliency which is rouge-1(tar,temp) + rouge-2(tar,temp) for each in batch.
  Note that the begin and end tokens are present as well.C
  """

  r_scores = []
  for template in templates:
    r1 = rougescore.rouge_n(target, [template], 1, 0.5)
    r2 = rougescore.rouge_n(target, [template], 2, 0.5)
    r_scores.append(r1+r2)

  return r_scores

def sort_by_length(input: list) -> tuple:
  forward = list(reversed(sorted(range(len(input)), key=lambda i: len(input[i]))))
  sorted_input = np.array(input)[forward]

  # Determine reverse index to restore original order
  backward = torch.empty(len(forward)).long()
  for i, j in enumerate(forward):
    backward[j] = i

  return sorted_input, backward, forward

class sentences_saver:
  """
  This class is created since the use of online data, adding each individually is very slow. 
  Saving them in memory and writing at the end is quicker.
  """
  
  def __init__(self, filename, split_token='\n'):
    self.data = []
  
    self.filename = filename
    self.split_token = split_token
    
  def store_sentence(self, sentence):
    self.data.append(sentence)
      
  def write_to_file(self):
    with open(self.filename, 'a+') as f:
      for sentence in self.data:
        f.write(sentence)
        f.write(self.split_token)
    self.data = []
    
    
def make_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)
    
def save_word2id(P, word2id: Word2Id):
  filename = '{}/word2id-{}.pi'.format(P.WORD2ID_DIR, P.MIN_OCCURENCE)
  print("Storing word2id as {}\n".format(filename))  
  
  word2id.w2id = dict(word2id.w2id) # transform in normal dict to store. using pickle.
  
  filehandler = open(filename, 'wb')
  pickle.dump(word2id, filehandler)
  
def load_word2id(P):
  filename = '{}/word2id-{}.pi'.format(P.WORD2ID_DIR, P.MIN_OCCURENCE)
  print("Loading word2id from {}\n".format(filename))  
  filehandler = open(filename, 'rb')
  word2id = pickle.load(filehandler)
  return word2id

def does_word2id_exist(P):
  filename = '{}/word2id-{}.pi'.format(P.WORD2ID_DIR, P.MIN_OCCURENCE)
  return os.path.isfile(filename)