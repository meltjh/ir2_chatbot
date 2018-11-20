import torch
import torch.nn as nn
from time import time
from models import Model, DecoderRNN#, DecoderFullSentences
from read_data import get_datasets

# Params
NUM_EPOCHS = 10
BATCH_SIZE = 3#5
HIDDEN_DIM = 250#7#250
EMBEDDING_DIM = 250#11#250
MERGE_TYPE = "all"

print("Merge type: {}, epochs: {}, batch size: {}, hidden dim: {}, embedding dim: {}.".format(MERGE_TYPE, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM, EMBEDDING_DIM))

# Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data, _, _, word2id = get_datasets("data/main_data", BATCH_SIZE, MERGE_TYPE)
vocab_size = len(word2id.id2w)
# Model
model = Model(word2id, HIDDEN_DIM, EMBEDDING_DIM).to(device)
decoder = DecoderRNN(vocab_size, 4*HIDDEN_DIM, vocab_size).to(device)

#decoder_full_sentences = DecoderFullSentences(vocab_size, 4*HIDDEN_DIM, vocab_size, device)
bilinear_loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters())

criterion = nn.L1Loss()

# Initialize the bos and eos embeddings.
tag_bos_emb = torch.zeros(1, 1, vocab_size).to(device)
tag_bos_emb[0,0,word2id.tag_id_bos] = 1
#tag_eos_emb = torch.zeros(vocab_size, 1)
#tag_eos_emb[0,0,word2id.tag_id_eos] = 1

# Training loop
for epoch in range(NUM_EPOCHS):
  start_time = time()
  for batch_num, batch in enumerate(train_data):
    # Get batch
    input, target, templates, _ = tuple(batch)
    input = [torch.tensor(i).long().to(device) for i in input]
    target = [torch.tensor(i).long().to(device) for i in target]

    
    templates = list(map(lambda l: [torch.tensor(i).long().to(device) for i in l], templates))
    # print(list(map(lambda l: [len(i) for i in l], templates)))

    # Training step
    opt.zero_grad()
    saliency, output_states = model(input, templates)
#    bilinear_loss, output_states = torch.stack([bilinear_loss_fn(sal, torch.randn_like(sal).float()) for sal in saliency]).mean()
    
    # target: num_batches x sentence_len with word id's

    train = True
    ########################################################################
    loss = 0
    for batch_i in range(output_states.shape[0]):
      os = output_states[batch_i]
      if train:
        prev_h = os.unsqueeze(0)
        prev_h = prev_h.unsqueeze(0)
        prev_output = tag_bos_emb
        for target_word_id in target[batch_i]:
          output, prev_h = decoder(prev_output, prev_h)
          t = torch.zeros(1, len(word2id.id2w)).to(device)
          t[0,target_word_id] = 1
          c = criterion(t, output)
          loss += c
          prev_output = output
          
#      else: # test: # TODO
#        sentence = []
#        prev_h = output_states
#        for i in range(max_len):
#          output, prev_h = self.decoder(prev_output, prev_h)
#          
#          sentence.append(output)
#          
#          if output == eos:
#            break
#          
#          prev_output = output
    
    ########################################################################
    # Initialize the bos and eos embeddings.
    
#    decoder_full_sentences(word2id.tag_id_bos, output_states, target)
    
    
    
    ########################################################################
    

    
    loss.backward()
    opt.step()

    # Progress
    print('\rEpoch {:03d}/{:03d} Example {:05d}/{:05d} ({:02d}:{:02d}/{:02d}:{:02d}) | Bilinear loss: {:.2f}'.format(
      epoch+1,
      NUM_EPOCHS,
      (batch_num+1)*BATCH_SIZE,
      len(train_data)*BATCH_SIZE,
      int((time()-start_time) // 60),
      int((time()-start_time) % 60),
      int(((time()-start_time)/(batch_num+1))*(len(train_data)-(batch_num)) // 60),
      int(((time()-start_time)/(batch_num+1))*(len(train_data)-(batch_num)) % 60),
      loss.item()
    ), end='')
