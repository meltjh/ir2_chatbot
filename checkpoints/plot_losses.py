import os
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def optain_all_data():
  """
  
  """
  
  # Obtain all folders
  folders = [f for f in os.listdir('.') if f != '__pycache__' and os.path.isdir(f)]
  
  # Process each checkpoint in the folders
  checkpoints_data = []
  for folder in folders:
    print('folder: {}'.format(folder))
    checkpoints = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]
    
    checkpoint_data = [(-1, -1, -1)] * len(checkpoints)
    for checkpoint in checkpoints:
      state = torch.load(checkpoint, map_location='cpu')
      
      epoch = state['epoch']
      if 'saliency_loss' in state or 'sailency_loss' in state or 'saiiency_loss' in state: # Note typos in key
            
        if 'saliency_loss' in state:
            saliency_loss = state['saliency_loss']
        elif 'sailency_loss' in state:
            saliency_loss = state['sailency_loss']
        elif 'saiiency_loss' in state:
            saliency_loss = state['saiiency_loss']
            
        decoder_loss = state['decoder_loss']
        if 'alpha' in state:
          alpha = state['alpha']
          total_loss = alpha*saliency_loss + (1-alpha)*decoder_loss
        else:
          total_loss = 0.5*saliency_loss + 0.5*decoder_loss
        
        checkpoint_data[epoch-1] = (saliency_loss, decoder_loss, total_loss)
      else:
        checkpoint_data[epoch-1] = (-1, -1, -1)
        
    checkpoints_data.append((folder, checkpoint_data))
  return checkpoints_data
    
def plot_all_data(checkpoints_data: list):
  """
  
  """
  legend_elements = [Line2D([0], [0], linestyle = '-.', color='black', alpha=.5, lw=2, label='Bilinear'),
                   Line2D([0], [0], linestyle = ':', color='black', alpha=.5, lw=2, label='Decoder'),
                   Line2D([0], [0], linestyle = '-', color='black', alpha=.5, lw=2, label='Total')]
  legend1 = plt.legend(handles=legend_elements, loc='upper left', fancybox=True, framealpha=0.5)
  
  ax = plt.gca()
  
  for checkpoint_name, checkpoint_data in checkpoints_data:
    x_indices = list(range(len(checkpoint_data)))
    saliency_loss, decoder_loss, total_loss = zip(*checkpoint_data)
    
    color = next(ax._get_lines.prop_cycler)['color']
    
    # compare_occ
#    if checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "9+ occurrence"
#    elif checkpoint_name == "batch64_hid256_emb100_meroracle_min500_bilinTrue_alpha0.5":
#      checkpoint_name = "500+ occurrence"
      
    # compare_alpha_batch
#    if checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
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
#    if checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "With bilinear"
#    elif checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinFalse_alpha0.5":
#      checkpoint_name = "Without bilinear"
    
    # compare_oracle_mixed
#    if checkpoint_name == "batch64_hid256_emb100_meroracle_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "Oracle"
#    elif checkpoint_name == "batch64_hid256_emb100_mermixed_min9_bilinTrue_alpha0.5":
#      checkpoint_name = "Mixed"    
    
    if max(total_loss) < 0:
      continue
    
    if max(saliency_loss) > 0:
      plt.plot(x_indices, saliency_loss, linestyle='-.', color=color, alpha=0.5)
      plt.plot(x_indices, decoder_loss, linestyle=':', color=color, alpha=0.5)
    plt.plot(x_indices, total_loss, label=checkpoint_name, linestyle='-', color=color, alpha=0.5)
  
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  
#  handles, labels = plt.gca().get_legend_handles_labels()
#  order = [4,2,1,0,3]
#  plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fancybox=True, framealpha=0.4, loc='upper right')
  
  plt.legend(fancybox=True, framealpha=0.4, loc='upper right')
  plt.gca().add_artist(legend1)
  plt.show()
  
checkpoints_data = optain_all_data()
plot_all_data(checkpoints_data)