import glob
from pythonrouge.pythonrouge.pythonrouge import Pythonrouge
#import bleu
import numpy as np
import matplotlib.pyplot as plt

#import sys
#f1 = sys.argv[1] #decoded
#f2 = sys.argv[2] #reference
#f3 = sys.argv[3] #highest index
f1 = "evaluation/result_data/True/input_ids_e"
f2 = "evaluation/result_data/True/response_ids_e" 
f3 = 9

#bl_scores = []
r1_r_scores = []
r2_r_scores = []
rL_r_scores = []
r1_f_scores = []
r2_f_scores = []
rL_f_scores = []

for idx in range(f3+1):
  ref = []
  dec = []

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
    dec.append(dec_tex)
  
  ref = [ref]
   
  # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
  # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
  # if recall_only=True, you can get recall scores of ROUGE
  print("\nRouge scores...")
  rouge = Pythonrouge(summary_file_exist=False,
                    summary=dec, reference=ref,
                    n_gram=2, ROUGE_SU4=False, ROUGE_L=False,
                    recall_only=False, stemming=False, stopwords=False,
                    word_level=True, length_limit=False, length=100,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=False, samples=1000, favor=False, p=0.5)
  score = rouge.calc_score()
  print(score)
  r1_r_scores.append(score['ROUGE-1-R'])
  r2_r_scores.append(score['ROUGE-2-R'])
#  rL_r_scores.append(score['ROUGE-L-R'])
  
  r1_f_scores.append(score['ROUGE-1-F'])
  r2_f_scores.append(score['ROUGE-2-F'])
#  rL_f_scores.append(score['ROUGE-L-F'])


x = list(range(1, f3+2))
#plt.plot(x, bl_scores, label="bl scores")
plt.plot(x, r1_r_scores, label="r1 R scores")
plt.plot(x, r2_r_scores, label="r2 R scores")
#plt.plot(x, rL_r_scores, label="rL R scores")
plt.plot(x, r1_f_scores, label="r1 F scores")
plt.plot(x, r2_f_scores, label="r2 F scores")
#plt.plot(x, rL_f_scores, label="rL F scores")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Score out of ?")
plt.show()