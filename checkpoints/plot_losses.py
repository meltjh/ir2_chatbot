import os
import torch
import matplotlib.pyplot as plt

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
  ax = plt.gca()
  
  for checkpoint_name, checkpoint_data in checkpoints_data:
    x_indices = list(range(len(checkpoint_data)))
    saliency_loss, decoder_loss, total_loss = zip(*checkpoint_data)
    
    color = next(ax._get_lines.prop_cycler)['color']
    
    if max(total_loss) < 0:
      continue
    
    if max(saliency_loss) > 0:
      plt.plot(x_indices, saliency_loss, label="{}  saliency".format(checkpoint_name), linestyle='-.', color=color, alpha=0.5)
      plt.plot(x_indices, decoder_loss, label="{}  decoder".format(checkpoint_name), linestyle=':', color=color, alpha=0.5)
    plt.plot(x_indices, total_loss, label="{}  total".format(checkpoint_name), linestyle='-', color=color, alpha=0.5)
  
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()
  
checkpoints_data = optain_all_data()
plot_all_data(checkpoints_data)