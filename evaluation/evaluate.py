"""
    Code from https://github.com/nikitacs16/Holl-E/
"""
import sys
import glob
import rouge
import bleu
import numpy as np
import matplotlib.pyplot as plt

fr = sys.argv[1] #folder
rg = int(sys.argv[2]) #range

f1 = "result_data/{}/input_ids_e".format(fr)
f2 = "result_data/{}/response_ids_e".format(fr)
f3 = rg


bl_scores = []
r1_scores = []
r2_scores = []

for idx in range(f3+1):
  ref = []
  decoded = []

  idx = str(idx)
  print("\n\n\nResults for index", idx)
  for i, j in zip(sorted(glob.glob(f1+idx+'*.txt')),sorted(glob.glob(f2+idx+'*.txt'))):
    ref_tex = ''
    dec_tex = ''
    for k in open(i).readlines():
      dec_tex = dec_tex + k.strip()
    for l in open(j).readlines():
      ref_tex = ref_tex + l.strip()
    ref.append(ref_tex)
    decoded.append(dec_tex)
  
  # Blue
  print("\nBlue score...")
  bl = bleu.moses_multi_bleu(decoded, ref)
  print(bl)
  bl_scores.append(bl)
  
  # Rouge 1
  print("\nRouge 1 score...")
  f1_score, precision, recall = rouge.rouge_n(decoded, ref, 1)
  print(f1_score*100)#, precision, recall)
  r1_scores.append(f1_score*100)
  
  # Rouge 2
  print("\nRouge 2 score...")
  f1_score, precision, recall = rouge.rouge_n(decoded, ref, 2)
  print(f1_score*100)#, precision, recall)
  r2_scores.append(f1_score*100)
  
  ## Rouge l
  #print("\nCalculating the rouge l score...")
  #f1_score, precision, recall = rouge.rouge_l_sentence_level(decoded, ref)
  #print(f1_score, precision, recall)


x = list(range(1, f3+2))
plt.plot(x, bl_scores, label="bl scores")
plt.plot(x, r1_scores, label="r1 scores")
plt.plot(x, r2_scores, label="r2 scores")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Score out of 100")
plt.title(fr)
plt.show()



#print("Calculating the rouge 2 score...")
#f1_score, precision, recall = rouge.rouge_n(decoded, ref, 2)
#print(f1_score, precision, recall)


#x = rouge.rouge(decoded,ref)
#
#print(bl)
#print(x['rouge_1/f_score'])
#print(x['rouge_2/f_score'])
#print(x['rouge_l/f_score'])
#
#print("blue: {:.2f}, rouge_1/f_score: {:.2f}, rouge_2/f_score: {:.2f}, rouge_l/f_score: {:.2f}".format(
#    bl,x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100))
##print('%.2f\t%.2f\t%.2f\t%.2f'%(bl,x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100))
#
