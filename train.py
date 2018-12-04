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
  BATCH_SIZE = 50
  HIDDEN_DIM = 256
  EMBEDDING_DIM = 100
  MERGE_TYPE = "oracle"
  SAVE_DIR = 'checkpoints'
  
def evaluate(postfix, data, model, word2id, decoder_loss_fn, saliency_loss_fn, device):
  """
  Evaluates and saves the generated sentences to the txt files. The postfix can be used for denoting the progress of training so far.
  """
  
  FOLDER = "evaluation/result_data/"
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
      
      # Write the results to txt files
      # Write the indices version.
      save_sentence_to_file(' '.join(str(e) for e in list(inp.cpu().numpy())), "{}input_ids_{}.txt".format(FOLDER, postfix))
      save_sentence_to_file(' '.join(str(e) for e in list(response.cpu().numpy())), "{}response_ids_{}.txt".format(FOLDER, postfix))
      
      # Write the string version.
#      save_sentence_to_file(word2id.id2string(inp), "{}input_str_{}.txt".format(FOLDER, postfix))
#      save_sentence_to_file(word2id.id2string(response), "{}response_str_{}.txt".format(FOLDER, postfix))
  
    print_progress("Evaluating: ",P, epoch, batch_num, len(data), saliency_loss, decoder_loss, start_time)
  print()


# %%
print("Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}.".format(P.MERGE_TYPE, P.NUM_EPOCHS, P.BATCH_SIZE, P.HIDDEN_DIM, P.EMBEDDING_DIM))

# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
glove_vocab, glove_embeddings = get_glove_embeddings(P.EMBEDDING_DIM)
train_data, val_data, _, word2id = get_datasets("data/experiment_data/bidaf/{}_short/".format(P.MERGE_TYPE), P.BATCH_SIZE, glove_vocab, False)
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


#    print("input", len(input))
    # Training step
    opt.zero_grad()
    saliency, response = model(input, target, templates)

    decoder_target = [t[1:] for t in target]  # Cut <BOS> from target
    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, decoder_target)]).mean()
    saliency_loss = torch.stack([saliency_loss_fn(sal, true_sal) for sal, true_sal in zip(saliency, target_saliencies)]).mean()

    (saliency_loss + decoder_loss).backward()
    opt.step()

    
#    for inp, templ in zip(input, templates):
#      response, _ = model.respond(device, word2id, [inp], [templ], max_length=20)
#      print('\nInput: \t\t {}'.format(word2id.id2string(inp)))
#      print('Response: \t {}'.format(word2id.id2string(response)))
    
    # Progress
    print_progress("Training: ", P, epoch, batch_num, len(train_data), saliency_loss.item(), decoder_loss.item(), start_time)

#    if batch_num != 0 and batch_num % 100 == 0:
#  postfix = "e{}_i{}".format(epoch,batch_num)
    postfix = "e{}".format(epoch)
  evaluate(postfix, val_data, model, word2id, decoder_loss_fn, saliency_loss_fn, device)
      
  save_checkpoint(P, epoch, model, opt)

# %%
# import importlib, models, utils
# importlib.reload(models)
# importlib.reload(utils)
# from models import Model
# from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch
