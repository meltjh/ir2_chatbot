import os
import matplotlib.pyplot as plt
import re
import rouge
import bleu

def sort_filenames_on_epoch(folder: str, starts_with:str):

  filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(starts_with) and f.endswith('.txt')]

  sorted_filenames = [None] * 10#len(filenames)
  for filename in filenames:
    epoch = int(re.findall(r"e{1}\d+\.", filename)[0][1:-1]) # Only get the epoch out of the string.
    sorted_filenames[epoch] = filename
  
  return sorted_filenames
  
  
def optain_all_data():
  """
  
  """
  
  # Obtain all folders
  folders = [f for f in os.listdir('.') if f != '__pycache__' and os.path.isdir(f)]
  
  # Process each checkpoint in the folders
  epochs_data = []
  for folder in folders:
    print('folder:{}'.format(folder))
    input_fname = os.path.join(folder, 'target_str.txt')
    sorted_fname_responses = sort_filenames_on_epoch(folder, 'response_str')

    epoch_data = []
    for i in range(len(sorted_fname_responses)):
      response_fname = sorted_fname_responses[i]

      if response_fname==None:
        epoch_data.append((-1,-1,-1))
        continue
        
      ref_tex = []
      dec_tex = []
      for k in open(input_fname).readlines():
        dec_tex.append(k.strip())
      for l in open(response_fname).readlines():
        ref_tex.append(l.strip())
    
#      print(dec_tex[0])
#      print(ref_tex[0])
#      print()
#      print()
      
      # Blue
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
  ax = plt.gca()
  
  for checkpoint_name, epoch_data in epochs_data:
    x_indices = list(range(len(epoch_data)))
    bl, r1_f1_score, r2_f1_score = zip(*epoch_data)
    
    color = next(ax._get_lines.prop_cycler)['color']
    
    plt.plot(x_indices, bl, label="{}  bleu".format(checkpoint_name), linestyle='-.', color=color, alpha=0.5)
    plt.plot(x_indices, r1_f1_score, label="{}  rouge 1 f1".format(checkpoint_name), linestyle=':', color=color, alpha=0.5)
    plt.plot(x_indices, r2_f1_score, label="{}  rouge 2 f1".format(checkpoint_name), linestyle='-', color=color, alpha=0.5)
  
  plt.legend()
  plt.show()
  
epochs_data = optain_all_data()
plot_all_data(epochs_data)