import torch
import torch.nn as nn
from time import time
from models import Model
from read_data2 import get_datasets

# Params
NUM_EPOCHS = 10
BATCH_SIZE = 10#3#5
HIDDEN_DIM = 250#7#250
EMBEDDING_DIM = 250#11#250
MERGE_TYPE = "oracle"

print("Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}.".format(MERGE_TYPE, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM, EMBEDDING_DIM))

# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data, _, _, word2id = get_datasets("data/experiment_data/bidaf/oracle_short/", BATCH_SIZE, False)
vocab_size = len(word2id.id2w)

# Model
model = Model(word2id, HIDDEN_DIM, EMBEDDING_DIM).to(device)

bilinear_loss_fn = nn.MSELoss()
decoder_loss_fn = nn.NLLLoss()
opt = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(NUM_EPOCHS):
  start_time = time()
  for batch_num, batch in enumerate(train_data):
    # Get batch
    input, target, templates, _ = tuple(batch)

    input = [torch.tensor(i).long().to(device) for i in input]
    target = [torch.tensor(i).long().to(device) for i in target]
    templates = list(map(lambda l: [torch.tensor(i).long().to(device) for i in l], templates))

    # Training step
    opt.zero_grad()
    saliency, response = model(input, target, templates)

    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, target)]).mean()
    bilinear_loss = torch.stack([bilinear_loss_fn(sal, torch.randn_like(sal).float()) for sal in saliency]).mean()

    (bilinear_loss + decoder_loss).backward()
    opt.step()

    # Progress
    print('\rEpoch {:03d}/{:03d} Example {:05d}/{:05d} ({:02d}:{:02d}/{:02d}:{:02d}) | Bilinear loss: {:.2f}, Decoder loss: {:.2f}'.format(
      epoch+1,
      NUM_EPOCHS,
      (batch_num+1)*BATCH_SIZE,
      len(train_data)*BATCH_SIZE,
      int((time()-start_time) // 60),
      int((time()-start_time) % 60),
      int(((time()-start_time)/(batch_num+1))*(len(train_data)-(batch_num)) // 60),
      int(((time()-start_time)/(batch_num+1))*(len(train_data)-(batch_num)) % 60),
      bilinear_loss.item(),
      decoder_loss.item()
    ), end='')
