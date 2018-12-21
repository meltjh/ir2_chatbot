import os
import re
import argparse
import torch
import torch.nn as nn
from time import time
from utils import print_progress, save_checkpoint, load_checkpoint, unpack_batch, sentences_saver, make_dir, save_word2id, load_word2id, does_word2id_exist
from models import Model
from read_data import get_single_dataset
from embeddings import get_glove_embeddings, get_embeddings_matrix



parser = argparse.ArgumentParser()

#parser.add_argument('--n_epochs', type=int, default=50,
#                    help='number of epochs')

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

args, _ = parser.parse_known_args()


# Params
class P:
  NUM_EPOCHS = 1
  BATCH_SIZE = args.batch_size
  HIDDEN_DIM = args.hidden_dim
  EMBEDDING_DIM = args.emb_dim
  MERGE_TYPE = args.merge_type
  MIN_OCCURENCE = args.min_occ
  USE_BILINEAR = (args.use_bilin == 'True') or (args.use_bilin == 'y')
  EXP_ID_PREFIX = (args.exp_id_prefix if args.exp_id_prefix != "" else "batch{}_hid{}_emb{}_mer{}_min{}_bilin{}".format(args.batch_size, args.hidden_dim, args.emb_dim, args.merge_type, args.min_occ, args.use_bilin)) 
  CHECKPOINT_DIR = 'checkpoints/{}/'.format(EXP_ID_PREFIX)
  EVAL_DIR = "evaluation/result_data/{}/".format(EXP_ID_PREFIX)
  WORD2ID_DIR = 'word2id/'
#  TARGET_DIR = "evaluation/targets/{}/".format(MIN_OCCURENCE)
  

make_dir(P.EVAL_DIR)





# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)

glove_vocab, glove_embeddings = get_glove_embeddings(P.EMBEDDING_DIM)

word2id = load_word2id(P)
data = get_single_dataset("data/experiment_data/bidaf/{}_short/{}-v1.1.json".format(P.MERGE_TYPE, "dev"), word2id, P.BATCH_SIZE, False, P.MIN_OCCURENCE, False, glove_vocab, False)
  
vocab_size = len(word2id.id2w)
embeddings_matrix = get_embeddings_matrix(glove_embeddings, word2id, vocab_size, P.EMBEDDING_DIM)



# %% Model
model = Model(word2id, P.HIDDEN_DIM, P.EMBEDDING_DIM, embeddings_matrix, P.USE_BILINEAR).to(device)

saliency_loss_fn = nn.MSELoss()
decoder_loss_fn = nn.NLLLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
opt = torch.optim.Adam(parameters)

  
def evaluate_checkpoint(P, postfix, data, word2id, checkpoint_fname, epoch, device):
  """
  
  """
  # Check for checkpoints
  _ = load_checkpoint(P, model, opt, device, checkpoint_fname)
    
  response_filename = "{}response_str_{}.txt".format(P.EVAL_DIR, postfix)
  if os.path.isfile(response_filename):
    print("Skipping {}, it does already exist.".format(response_filename))
    return
  saver_response_str = sentences_saver(response_filename)
  

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

    for inp, templ, targ in zip(input, templates, target):
      response, _ = model.respond(device, word2id, [inp], [templ], max_length=50)

      # Write the results to txt files
      saver_response_str.store_sentence(word2id.id2string(response))
      
    print_progress("Evaluating: ", P, epoch-1, batch_num, len(data), total_saliency_loss/(batch_num+1), total_decoder_loss/(batch_num+1), start_time)
    
  print()

  saver_response_str.write_to_file()
  
  
checkpoints = checkpoints = [os.path.join(P.CHECKPOINT_DIR, f) for f in os.listdir(P.CHECKPOINT_DIR) if f.endswith('.pt')]
for checkpoint_fname in checkpoints:
  epoch = int(re.findall(r"cp-{1}\d+\.", checkpoint_fname)[0][3:-1]) # Only get the epoch out of the string.
  evaluate_checkpoint(P, 'e{}'.format(epoch), data, word2id, checkpoint_fname, epoch, device)