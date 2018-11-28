import torch
import torch.nn as nn
from time import time
from models import Model
from read_data import get_datasets
from embeddings import get_glove_embeddings, get_embeddings_matrix

# Params
NUM_EPOCHS = 10
BATCH_SIZE = 5#5
HIDDEN_DIM = 50#250
EMBEDDING_DIM = 300#250
MERGE_TYPE = "oracle"

print("Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}.".format(MERGE_TYPE, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM, EMBEDDING_DIM))

# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
glove_vocab, glove_embeddings = get_glove_embeddings(EMBEDDING_DIM)
train_data, _, _, word2id = get_datasets("data/experiment_data/bidaf/oracle_short/", BATCH_SIZE, glove_vocab, False)
vocab_size = len(word2id.id2w)
embeddings_matrix = get_embeddings_matrix(glove_embeddings, word2id, vocab_size, EMBEDDING_DIM)

# %% Model
model = Model(word2id, HIDDEN_DIM, EMBEDDING_DIM, embeddings_matrix).to(device)
# model.embeddings.weight.data.copy_(torch.from_numpy(embeddings_matrix))

bilinear_loss_fn = nn.MSELoss()
decoder_loss_fn = nn.NLLLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
opt = torch.optim.Adam(parameters)

# Training loop
for epoch in range(NUM_EPOCHS):
  start_time = time()
  for batch_num, batch in enumerate(train_data):
    # Get batch
    input, target, templates, _, true_saliencies = tuple(batch)

    input = [torch.tensor(i).long().to(device) for i in input]
    target = [torch.tensor(i).long().to(device) for i in target]
    templates = list(map(lambda l: [torch.tensor(i).long().to(device) for i in l], templates))
    true_saliencies = [torch.tensor(i).float().to(device).unsqueeze(1) for i in true_saliencies]


    # Training step
    opt.zero_grad()
    saliency, response = model(input, target, templates)

    decoder_target = [t[1:] for t in target]  # Cut <BOS> from target
    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, decoder_target)]).mean()
    bilinear_loss = torch.stack([bilinear_loss_fn(sal, torch.randn_like(sal).float()) for sal in saliency]).mean()

    (bilinear_loss + decoder_loss).backward()
    opt.step()

    response, _ = model.respond(device, word2id, input[:1], templates[:1], max_length=20)
    print('\nInput: \t\t {}'.format(' '.join([word2id.id2w[x] for x in input[0]])))
    print('Response: \t {}'.format(' '.join([word2id.id2w[x] for x in response])))

    # Progress
    print('\rEpoch {:03d}/{:03d} Example {:05d}/{:05d} ({:02d}:{:02d}/{:02d}:{:02d}) | Bilinear loss: {:.4f}, Decoder loss: {:.4f}'.format(
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

# %%
# import importlib, models
# importlib.reload(models)
# from models import Model
