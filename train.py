import torch
import torch.nn as nn
from time import time
from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch, save_sentence_to_file
from models import Model
from read_data import get_datasets
from embeddings import get_glove_embeddings, get_embeddings_matrix



# Params
class P:
  NUM_EPOCHS = 10
  BATCH_SIZE = 3
  HIDDEN_DIM = 15
  EMBEDDING_DIM = 50
  MERGE_TYPE = "oracle"
  SAVE_DIR = 'checkpoints'
  
def evaluate(data, model, word2id, decoder_loss_fn, saliency_loss_fn, device):
  # TODO: Genereren per 1, niet batches.
  # Zinnen opslaan in doc voor evaluatie.
  print()
  total_decoder_loss = 0
  total_saliency_loss = 0
  
  start_time = time()
  for batch_num, batch in enumerate(data):
    # Get batch
    input, target, templates, target_saliencies = unpack_batch(batch, device)

    saliency, response = model(input, target, templates)

    decoder_target = [t[1:] for t in target]  # Cut <BOS> from target
    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, decoder_target)]).mean()
    saliency_loss = torch.stack([saliency_loss_fn(sal, true_sal) for sal, true_sal in zip(saliency, target_saliencies)]).mean()

    total_decoder_loss += decoder_loss.item()
    total_saliency_loss += saliency_loss.item()
    
    for inp, templ in zip(input, templates):
      response, _ = model.respond(device, word2id, [inp], [templ], max_length=20)
      str_input = word2id.id2string(inp)
      str_response = word2id.id2string(response)
#      print('\nInput: \t\t {}'.format(str_input))
#      print('Response: \t {}'.format(str_response))
      
      save_sentence_to_file(str_input, "input_str.txt")
      save_sentence_to_file(str_response, "response_str.txt")
      
      save_sentence_to_file(' '.join(str(e) for e in list(inp.numpy())), "input_idstxt")
      save_sentence_to_file(' '.join(str(e) for e in list(response.numpy())), "response_ids.txt")
      
    raise NotImplementedError("±±± ±±± ±±±")

    print_progress("Evaluating: ",P, epoch, batch_num, len(data), saliency_loss, decoder_loss, start_time)



# %%
print("Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}.".format(P.MERGE_TYPE, P.NUM_EPOCHS, P.BATCH_SIZE, P.HIDDEN_DIM, P.EMBEDDING_DIM))

# Init
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#glove_vocab, glove_embeddings = get_glove_embeddings(P.EMBEDDING_DIM)
#train_data, val_data, _, word2id = get_datasets("data/experiment_data/bidaf/oracle_short/", P.BATCH_SIZE, glove_vocab, False)
#vocab_size = len(word2id.id2w)
#embeddings_matrix = get_embeddings_matrix(glove_embeddings, word2id, vocab_size, P.EMBEDDING_DIM)

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


#    print("input", len(input))
    # Training step
    opt.zero_grad()
    saliency, response = model(input, target, templates)

    decoder_target = [t[1:] for t in target]  # Cut <BOS> from target
    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, decoder_target)]).mean()
    saliency_loss = torch.stack([saliency_loss_fn(sal, true_sal) for sal, true_sal in zip(saliency, target_saliencies)]).mean()

    (saliency_loss + decoder_loss).backward()
    opt.step()

    
    # TODO: Check of de input zinnen wel kloppen..
#    for inp, templ in zip(input, templates):
#      response, _ = model.respond(device, word2id, [inp], [templ], max_length=20)
#      print('\nInput: \t\t {}'.format(word2id.id2string(inp)))
#      print('Response: \t {}'.format(word2id.id2string(response)))
    
    # Progress
    print_progress("Training: ", P, epoch, batch_num, len(train_data), saliency_loss.item(), decoder_loss.item(), start_time)

#    if batch_num % 100 == 0:
#      evaluate(val_data, model, word2id, decoder_loss_fn, saliency_loss_fn, device)
  save_checkpoint(P, epoch, model, opt)

# %%
# import importlib, models, utils
# importlib.reload(models)
# importlib.reload(utils)
# from models import Model
# from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch
