import torch
import torch.nn as nn
from time import time
from models import Model
from read_data import get_datasets
from embeddings import get_glove_embeddings, get_embeddings_matrix
from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch


# Params
class P:
  NUM_EPOCHS = 10
  BATCH_SIZE = 5
  HIDDEN_DIM = 50
  EMBEDDING_DIM = 300
  MERGE_TYPE = "oracle"
  SAVE_INTERVAL = 10000
  SAVE_DIR = 'checkpoints'


# %%
print("Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}.".format(P.MERGE_TYPE, P.NUM_EPOCHS, P.BATCH_SIZE, P.HIDDEN_DIM, P.EMBEDDING_DIM))

# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
glove_vocab, glove_embeddings = get_glove_embeddings(P.EMBEDDING_DIM)
train_data, _, _, word2id = get_datasets("data/experiment_data/bidaf/oracle_short/", P.BATCH_SIZE, glove_vocab, False)
vocab_size = len(word2id.id2w)
embeddings_matrix = get_embeddings_matrix(glove_embeddings, word2id, vocab_size, P.EMBEDDING_DIM)

# %% Model
model = Model(word2id, P.HIDDEN_DIM, P.EMBEDDING_DIM, embeddings_matrix).to(device)

saliency_loss_fn = nn.MSELoss()
decoder_loss_fn = nn.NLLLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
opt = torch.optim.Adam(parameters)

# Check for checkpoints
start_epoch = load_checkpoint(P, model, opt)

# Training loop
for epoch in range(start_epoch, P.NUM_EPOCHS):
  start_time = time()
  for batch_num, batch in enumerate(train_data):
    # Get batch
    input, target, templates, target_saliencies = unpack_batch(batch, device)

    # Training step
    opt.zero_grad()
    saliency, response = model(input, target, templates)

    decoder_target = [t[1:] for t in target]  # Cut <BOS> from target
    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, decoder_target)]).mean()
    saliency_loss = torch.stack([saliency_loss_fn(sal, true_sal) for sal, true_sal in zip(saliency, target_saliencies)]).mean()

    (saliency_loss + decoder_loss).backward()
    opt.step()

    response, _ = model.respond(device, word2id, input[:1], templates[:1], max_length=20)
    print('\nInput: \t\t {}'.format(' '.join([word2id.id2w[x] for x in input[0]])))
    print('Response: \t {}'.format(' '.join([word2id.id2w[x] for x in response])))

    # Progress
    print_progress(P, epoch, batch_num, len(train_data), saliency_loss, decoder_loss, start_time)

  save_checkpoint(P, epoch, model, opt)

# %%
# import importlib, models, utils
# importlib.reload(models)
# importlib.reload(utils)
# from models import Model
# from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch
