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

#parser.add_argument('--n_epochs', type=int, default=50,
#                    help='number of epochs')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')

parser.add_argument('--hidden_dim', type=int, default=128,
                    help='dimensionality of the hidden space')

parser.add_argument('--emb_dim', type=int, default=100,
                    help='dimensionality of the word embeddings')

#parser.add_argument('--merge_type', type=str, default='oracle',
#                    help='dataset')

parser.add_argument('--min_occ', type=int, default=500,
                    help='minimal amount of occurences for a word to be used')

parser.add_argument('--use_bilin', type=str, default='True',
                    help='is the bilinear part used')

args, _ = parser.parse_known_args()


# Params
class P:
#  NUM_EPOCHS = args.n_epochs
  BATCH_SIZE = args.batch_size
  HIDDEN_DIM = args.hidden_dim
  EMBEDDING_DIM = args.emb_dim
#  MERGE_TYPE = args.merge_type
  MIN_OCCURENCE = args.min_occ
  USE_BILINEAR = (args.use_bilin == 'True') or (args.use_bilin == 'y')
  EXP_ID_PREFIX = args.exp_id_prefix
  CHECKPOINT_DIR = 'checkpoints/{}_{}_{}/'.format(EXP_ID_PREFIX, MIN_OCCURENCE, USE_BILINEAR)
  EVAL_DIR = "evaluation/result_data/{}_{}_{}/".format(EXP_ID_PREFIX, MIN_OCCURENCE, USE_BILINEAR)
#  TARGET_DIR = "evaluation/targets/{}/".format(MIN_OCCURENCE)
  

#make_dir(P.SAVE_DIR)
make_dir(P.EVAL_DIR)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)

glove_vocab, glove_embeddings = get_glove_embeddings(P.EMBEDDING_DIM)
_, dev_data, _, word2id = get_datasets("data/experiment_data/bidaf/{}_short/".format(P.MERGE_TYPE), P.BATCH_SIZE, P.MIN_OCCURENCE, glove_vocab, False, True)
vocab_size = len(word2id.id2w)
embeddings_matrix = get_embeddings_matrix(glove_embeddings, word2id, vocab_size, P.EMBEDDING_DIM)

  # %% Model
  model = Model(word2id, P.HIDDEN_DIM, P.EMBEDDING_DIM, embeddings_matrix, P.USE_BILINEAR).to(device)
  
  saliency_loss_fn = nn.MSELoss()
  decoder_loss_fn = nn.NLLLoss()
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  opt = torch.optim.Adam(parameters)
  
  # Load the checkpoint
  load_checkpoint(P, model, opt, checkpoint_fname)
  
def evaluate_checkpoint(postfix, data, word2id, checkpoint_fname, device):
  """
  
  """
  
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

    for inp, templ, targ in zip(input, templates, target):
      response, _ = model.respond(device, word2id, [inp], [templ], max_length=20)

      # Write the results to txt files
      saver_response_str.store_sentence(word2id.id2string(response))

    print_progress("Evaluating: ", P, epoch, batch_num, len(data), total_saliency_loss/(batch_num+1), total_decoder_loss/(batch_num+1), start_time)
  print()
  saver_response_str.write_to_file()
  
  
evaluate_checkpoint(postfix, dev_data, word2id, checkpoint_fname, device)