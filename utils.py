import os
import torch
from torch import Tensor
from torch.optim import Optimizer

from time import time
from models import Model

def print_progress(P, epoch: int, batch_num: int, num_batches: int, saliency_loss: Tensor, decoder_loss: Tensor, start_time: float) -> None:
  elapsed_time = time() - start_time
  average_rate = elapsed_time / (batch_num + 1)
  remaining_time = average_rate * (num_batches - batch_num)

  print('\rEpoch {:03d}/{:03d} Example {:05d}/{:05d} ({:02d}:{:02d}/{:02d}:{:02d}) | Bilinear loss: {:.4f}, Decoder loss: {:.4f}'.format(
    epoch+1,
    P.NUM_EPOCHS,
    (batch_num+1)*P.BATCH_SIZE,
    num_batches*P.BATCH_SIZE,
    int(elapsed_time // 60),
    int(elapsed_time % 60),
    int(remaining_time // 60),
    int(remaining_time % 60),
    saliency_loss.item(),
    decoder_loss.item()
  ), end='')

def unpack_batch(batch: list, device: torch.device) -> tuple:
  input, target, templates, _, saliencies = tuple(batch)

  input = [torch.tensor(i).long().to(device) for i in input]
  target = [torch.tensor(i).long().to(device) for i in target]
  templates = list(map(lambda l: [torch.tensor(i).long().to(device) for i in l], templates))
  saliencies = [torch.tensor(i).float().to(device).unsqueeze(1) for i in saliencies]

  return input, target, templates, saliencies


def save_checkpoint(P, epoch: int, model: Model, optimiser: Optimizer) -> None:
  if not os.path.exists(P.SAVE_DIR):
    os.mkdir(P.SAVE_DIR)

  state = {
    'epoch': epoch + 1,
    'model': model.state_dict(),
    'optimiser': optimiser.state_dict()
  }

  torch.save(state, '{}/checkpoint-{}.pt'.format(P.SAVE_DIR, epoch+1))

def load_checkpoint(P, model: Model, optimiser: Optimizer) -> int:
  epoch = 0
  if os.path.exists(P.SAVE_DIR):
    checkpoints = [os.path.join(P.SAVE_DIR, f) for f in os.listdir(P.SAVE_DIR) if f.endswith('.pt')]

    if checkpoints:
      latest_checkpoint = max(checkpoints, key=os.path.getctime)
      state = torch.load(latest_checkpoint)

      model.load_state_dict(state['model'])
      optimiser.load_state_dict(state['optimiser'])
      epoch = state['epoch']

  return epoch
