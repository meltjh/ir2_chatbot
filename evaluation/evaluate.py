import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
import rouge
import bleu

def sort_filenames_on_epoch(folder: str, starts_with:str):

  filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(starts_with) and f.endswith('.txt')]

  sorted_filenames = [None] * len(filenames)
  for filename in filenames:
    epoch = int(re.findall(r"e{1}\d+\.", filename)[0][1:-1]) # Only get the epoch out of the string.
    sorted_filenames[epoch-1] = filename
  
  return sorted_filenames
  
  
def optain_all_data():
  """
  
  """
  main_folder = './result_data/'
  # Obtain all folders
  folders = [f for f in os.listdir(main_folder) if f != '__pycache__' and os.path.isdir(os.path.join(main_folder,f))]

  # Process each checkpoint in the folders
  epochs_data = []
  for folder in folders:
    print('folder:{}'.format(folder))
    input_fname = os.path.join('../data/tokenized_target.txt')
    sorted_fname_responses = sort_filenames_on_epoch(os.path.join(main_folder,folder), 'response_str')

    epoch_data = []
    for i in range(len(sorted_fname_responses)):
      response_fname = sorted_fname_responses[i]

      if response_fname==None:
        epoch_data.append((-1,-1,-1))
        continue
        
      ref_tex = []
      dec_tex = []
      for k in open(input_fname).readlines():
        sentence = k.strip()
        sentence = sentence.replace("<bos> ", "").replace(" <eos>", "")
        dec_tex.append(sentence)
      for l in open(response_fname).readlines():
        sentence = l.strip()
        sentence = sentence.replace("<bos> ", "").replace(" <eos>", "")
        ref_tex.append(sentence)
      
      # Bleu
      print("\nBleu score...")
      bl = bleu.moses_multi_bleu(dec_tex, ref_tex)
      print(bl)
      
      # Rouge 1
      print("\nRouge 1 score...")
      r1_f1_score, r1_precision, r1_recall = rouge.rouge_n(dec_tex, ref_tex, 1)
      print(r1_f1_score*100)#, precision, recall)
      
      # Rouge 2
      print("\nRouge 2 score...")
      r2_f1_score, r2_precision, r2_recall = rouge.rouge_n(dec_tex, ref_tex, 2)
      print(r2_f1_score*100)#, precision, recall)
     
    
#      # Rouge l
#      print("\nCalculating the rouge l score...")
#      f1_score, precision, recall = rouge.rouge_l_sentence_level(dec_tex, ref_tex)
#      print(f1_score*100)#, precision, recall)
      
      
      epoch_data.append((bl, r1_f1_score*100, r2_f1_score*100))
    
      
      
    epochs_data.append((folder, epoch_data))
  return epochs_data
    
def plot_all_data(epochs_data: list):
  """
  
  """
  legend_elements = [Line2D([0], [0], linestyle = '-.', color='black', alpha=.5, lw=2, label='BLEU'),
                   Line2D([0], [0], linestyle = ':', color='black', alpha=.5, lw=2, label='ROUGE 1'),
                   Line2D([0], [0], linestyle = '-', color='black', alpha=.5, lw=2, label='ROUGE 2')]
  legend1 = plt.legend(handles=legend_elements, loc='upper left', fancybox=True, framealpha=0.5)
  
  ax = plt.gca()
  
  for checkpoint_name, epoch_data in epochs_data:
    x_indices = list(range(len(epoch_data)))
    bl, r1_f1_score, r2_f1_score = zip(*epoch_data)
    
    color = next(ax._get_lines.prop_cycler)['color']
    
    # compare_occ
#    if checkpoint_name == "_batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "9+ occurrence"
#    elif checkpoint_name == "batch64_hid256_emb100_meroracle_min500_bilinTrue_alpha0.5":
#      checkpoint_name = "500+ occurrence"
      
    # compare_alpha_batch
#    if checkpoint_name == "_batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "Batch 64, alpha 0.5"
#    elif checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.7":
#      checkpoint_name = "Batch 64, alpha 0.7"
#    elif checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.99":
#      checkpoint_name = "Batch 64, alpha 0.99"
#    elif checkpoint_name == "batch4_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "Batch 4, alpha 0.5"
#    elif checkpoint_name == "batch4_hid256_emb100_meroracle_min9_bilinTrue_alpha0.7":
#      checkpoint_name = "Batch 4, alpha 0.7"

    # compare_bilinear
#    if checkpoint_name == "_batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "With bilinear"
#    elif checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinFalse_alpha0.5":
#      checkpoint_name = "Without bilinear"
    
    # compare_oracle_mixed
#    if checkpoint_name == "_batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "Oracle"
#    elif checkpoint_name == "batch64_hid256_emb100_mermixed_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "Mixed"
      
    plt.plot(x_indices, bl, linestyle='-.', color=color, alpha=0.5)
    plt.plot(x_indices, r1_f1_score, linestyle=':', color=color, alpha=0.5)
    plt.plot(x_indices, r2_f1_score, label=checkpoint_name, linestyle='-', color=color, alpha=0.5)
  
  plt.xlabel("Epochs")
  plt.ylabel("Scores")
  
#  handles, labels = plt.gca().get_legend_handles_labels()
#  order = [4,2,1,0,3]
#  plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fancybox=True, framealpha=0.4, loc='upper right')

  plt.legend(fancybox=True, framealpha=0.4, loc='upper right')
  plt.gca().add_artist(legend1)
  plt.show()
  
epochs_data = optain_all_data()
plot_all_data(epochs_data)