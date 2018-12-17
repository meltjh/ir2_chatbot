import argparse
import torch
import torch.nn as nn
from time import time
from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch, sentences_saver, make_dir
from models import Model
from read_data import get_datasets
from embeddings import get_glove_embeddings, get_embeddings_matrix



parser = argparse.ArgumentParser()

parser.add_argument('--exp_id_prefix', type=str, default="default",
                    help='the previx of this experiment')

parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of epochs')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')

parser.add_argument('--hidden_dim', type=int, default=128,
                    help='dimensionality of the hidden space')

parser.add_argument('--emb_dim', type=int, default=100,
                    help='dimensionality of the word embeddings')

parser.add_argument('--merge_type', type=str, default='oracle',
                    help='dataset')

parser.add_argument('--min_occ', type=int, default=500,
                    help='minimal amount of occurences for a word to be used')

parser.add_argument('--use_bilin', type=bool, default=True,
                    help='is the bilinear part used') # Note that a '' should be given for False. All strings are True.

args, _ = parser.parse_known_args()


# Params
class P:
  NUM_EPOCHS = args.n_epochs
  BATCH_SIZE = args.batch_size
  HIDDEN_DIM = args.hidden_dim
  EMBEDDING_DIM = args.emb_dim
  MERGE_TYPE = args.merge_type
  MIN_OCCURENCE = args.min_occ
  USE_BILINEAR = args.use_bilin
  EXP_ID_PREFIX = args.exp_id_prefix
  SAVE_DIR = 'checkpoints/{}_{}_{}/'.format(EXP_ID_PREFIX, MIN_OCCURENCE, USE_BILINEAR)
  FOLDER = "evaluation/result_data/{}_{}_{}/".format(EXP_ID_PREFIX, MIN_OCCURENCE, USE_BILINEAR)

make_dir(P.SAVE_DIR)
make_dir(P.FOLDER)

def evaluate(postfix, data, model, word2id, decoder_loss_fn, saliency_loss_fn, device):
  """
  Evaluates and saves the generated sentences to the txt files. The postfix can be used for denoting the progress of training so far.
  """
  
  saver_input_ids = sentences_saver("{}input_ids_{}.txt".format(P.FOLDER, postfix))
  saver_response_ids = sentences_saver("{}response_ids_{}.txt".format(P.FOLDER, postfix))
  saver_input_str = sentences_saver("{}input_str_{}.txt".format(P.FOLDER, postfix))
  saver_response_str = sentences_saver("{}response_str_{}.txt".format(P.FOLDER, postfix))

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

    if P.USE_BILINEAR:
      # Only when the bilinear is used, there is a sailency loss.
      saliency_loss = torch.stack([saliency_loss_fn(sal, true_sal) for sal, true_sal in zip(saliency, target_saliencies)]).mean()
      total_saliency_loss += saliency_loss.item()
    total_decoder_loss += decoder_loss.item()

    for inp, templ in zip(input, templates):
      response, _ = model.respond(device, word2id, [inp], [templ], max_length=20)

      # Write the results to txt files
      # Write the indices version.
      saver_input_ids.store_sentence(' '.join(str(e) for e in list(inp.cpu().numpy())))
      saver_response_ids.store_sentence(' '.join(str(e) for e in list(response.cpu().numpy())))

      # Write the string version.
      saver_input_str.store_sentence(word2id.id2string(inp))
      saver_response_str.store_sentence(word2id.id2string(response))

    print_progress("Evaluating: ", P, epoch, batch_num, len(data), total_saliency_loss/(batch_num+1), total_decoder_loss/(batch_num+1), start_time)
  print()
  saver_input_ids.write_to_file()
  saver_response_ids.write_to_file()
  saver_input_str.write_to_file()
  saver_response_str.write_to_file()


# %%
print("Bilinear: {}, Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}, min occurences: {}.".format(P.USE_BILINEAR, P.MERGE_TYPE, P.NUM_EPOCHS, P.BATCH_SIZE, P.HIDDEN_DIM, P.EMBEDDING_DIM, P.MIN_OCCURENCE))

# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
glove_vocab, glove_embeddings = get_glove_embeddings(P.EMBEDDING_DIM)
train_data, val_data, _, word2id = get_datasets("data/experiment_data/bidaf/{}_short/".format(P.MERGE_TYPE), P.BATCH_SIZE, P.MIN_OCCURENCE, glove_vocab, False)
vocab_size = len(word2id.id2w)
embeddings_matrix = get_embeddings_matrix(glove_embeddings, word2id, vocab_size, P.EMBEDDING_DIM)

# %% Model
model = Model(word2id, P.HIDDEN_DIM, P.EMBEDDING_DIM, embeddings_matrix, P.USE_BILINEAR).to(device)

saliency_loss_fn = nn.MSELoss()
decoder_loss_fn = nn.NLLLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
opt = torch.optim.Adam(parameters)

# Check for checkpoints
start_epoch = load_checkpoint(P, model, opt)

# Training loop
for epoch in range(start_epoch, P.NUM_EPOCHS):
  start_time = time()
  epoch_total_sailency_loss = 0
  epoch_total_decoder_loss = 0
  for batch_num, batch in enumerate(train_data):
    # Get batch
    input, target, templates, target_saliencies = unpack_batch(batch, device)


#    print("input", len(input))
    # Training step
    opt.zero_grad()
    saliency, response = model(input, target, templates)

    decoder_target = [t[1:] for t in target]  # Cut <BOS> from target
    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, decoder_target)]).mean()

    if P.USE_BILINEAR:
      saliency_loss = torch.stack([saliency_loss_fn(sal, true_sal) for sal, true_sal in zip(saliency, target_saliencies)]).mean()
      epoch_total_sailency_loss += saliency_loss.item()
      epoch_total_decoder_loss += decoder_loss.item()
      (saliency_loss + decoder_loss).backward()
    else:
      decoder_loss.backward()
      epoch_total_decoder_loss += decoder_loss.item()
    opt.step()


#    for inp, templ in zip(input, templates):
#      response, _ = model.respond(device, word2id, [inp], [templ], max_length=20)
#      print('\nInput: \t\t {}'.format(word2id.id2string(inp)))
#      print('Response: \t {}'.format(word2id.id2string(response)))

    # Progress
    print_progress("Training: ", P, epoch, batch_num, len(train_data), epoch_total_sailency_loss/(batch_num+1), epoch_total_decoder_loss/(batch_num+1), start_time)

#    if batch_num != 0 and batch_num % 100 == 0:
#  postfix = "e{}_i{}".format(epoch,batch_num)


  save_checkpoint(P, epoch, model, opt)
  postfix = "e{}".format(epoch)
  evaluate(postfix, val_data, model, word2id, decoder_loss_fn, saliency_loss_fn, device)

  

# %%
# import importlib, models, utils
# importlib.reload(models)
# importlib.reload(utils)
# from models import Model
# from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch


# %%
  


