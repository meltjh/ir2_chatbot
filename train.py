import argparse
import torch
import torch.nn as nn
from time import time
from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch, sentences_saver, make_dir, save_word2id, load_word2id, does_word2id_exist
from models import Model
from read_data import get_single_dataset
from embeddings import get_glove_embeddings, get_embeddings_matrix
from word2id import Word2Id

parser = argparse.ArgumentParser()

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

parser.add_argument('--use_bilin', type=str, default='True',
                    help='is the bilinear part used')

parser.add_argument('--exp_id_prefix', type=str, default="",
                    help='the previx of this experiment')

parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha*bilinear + (1-alpha)*decoder')

args, _ = parser.parse_known_args()


# Params
class P:
  NUM_EPOCHS = args.n_epochs
  BATCH_SIZE = args.batch_size
  HIDDEN_DIM = args.hidden_dim
  EMBEDDING_DIM = args.emb_dim
  MERGE_TYPE = args.merge_type
  MIN_OCCURENCE = args.min_occ
  USE_BILINEAR = (args.use_bilin == 'True') or (args.use_bilin == 'y')
  ALPHA = (args.alpha if USE_BILINEAR == True else 0) # Ignore the alpha if no bilinear is used.
  EXP_ID_PREFIX = (args.exp_id_prefix if args.exp_id_prefix != "" else "batch{}_hid{}_emb{}_mer{}_min{}_bilin{}_alpha{}".format(args.batch_size, args.hidden_dim, args.emb_dim, args.merge_type, args.min_occ, args.use_bilin, args.alpha)) 
  CHECKPOINT_DIR = 'checkpoints/{}/'.format(EXP_ID_PREFIX)
  WORD2ID_DIR = 'word2id/'

make_dir(P.CHECKPOINT_DIR)
make_dir(P.WORD2ID_DIR)


# %%
print("Bilinear: {}, Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}, min occurences: {}, alpha: {}.".format(P.USE_BILINEAR, P.MERGE_TYPE, P.NUM_EPOCHS, P.BATCH_SIZE, P.HIDDEN_DIM, P.EMBEDDING_DIM, P.MIN_OCCURENCE, P.ALPHA))

# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)

glove_vocab, glove_embeddings = get_glove_embeddings(P.EMBEDDING_DIM)

if does_word2id_exist(P) == False:
  
  word2id = Word2Id()
  train_data = get_single_dataset("data/experiment_data/bidaf/{}_short/{}-v1.1.json".format(P.MERGE_TYPE, "train"), word2id, P.BATCH_SIZE, True, P.MIN_OCCURENCE, True, glove_vocab, False)
  save_word2id(P, word2id)
  
else:
  word2id = load_word2id(P)
  train_data = get_single_dataset("data/experiment_data/bidaf/{}_short/{}-v1.1.json".format(P.MERGE_TYPE, "train"), word2id, P.BATCH_SIZE, False, P.MIN_OCCURENCE, True, glove_vocab, False)
  
vocab_size = len(word2id.id2w)
embeddings_matrix = get_embeddings_matrix(glove_embeddings, word2id, vocab_size, P.EMBEDDING_DIM)


# %% Model
model = Model(word2id, P.HIDDEN_DIM, P.EMBEDDING_DIM, embeddings_matrix, P.USE_BILINEAR).to(device)

saliency_loss_fn = nn.MSELoss()
decoder_loss_fn = nn.NLLLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
opt = torch.optim.Adam(parameters)

# Check for checkpoints
start_epoch = load_checkpoint(P, model, opt, device)

# Training loop
for epoch in range(start_epoch, P.NUM_EPOCHS):
  start_time = time()
  epoch_total_saliency_loss = 0
  epoch_total_decoder_loss = 0
  epoch_total_loss = 0
  for batch_num, batch in enumerate(train_data):
    # Get batch
    input, target, templates, target_saliencies = unpack_batch(batch, device)

    # Training step
    opt.zero_grad()
    saliency, response = model(input, target, templates)

    decoder_target = [t[1:] for t in target]  # Cut <BOS> from target
    decoder_loss = torch.stack([decoder_loss_fn(res, tar) for res, tar in zip(response, decoder_target)]).mean()

    if P.USE_BILINEAR:
      saliency_loss = torch.stack([saliency_loss_fn(sal, true_sal) for sal, true_sal in zip(saliency, target_saliencies)]).mean()
      loss = (args.alpha*saliency_loss + (1-args.alpha)*decoder_loss)
      loss.backward()
      
      epoch_total_saliency_loss += saliency_loss.item()
      epoch_total_decoder_loss += decoder_loss.item()
      epoch_total_loss += loss
    else:
      decoder_loss.backward()
      epoch_total_decoder_loss += decoder_loss.item()
    opt.step()

    # Progress
    print_progress("Training: ", P, epoch, batch_num, len(train_data), epoch_total_saliency_loss/(batch_num+1), epoch_total_decoder_loss/(batch_num+1), start_time)

  save_checkpoint(P, epoch, model, opt, epoch_total_saliency_loss/(batch_num+1), epoch_total_decoder_loss/(batch_num+1))

