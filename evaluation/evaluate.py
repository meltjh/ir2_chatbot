"""
    Code from https://github.com/nikitacs16/Holl-E/
"""

import sys
import glob
import rouge
import bleu
f1 = "result_data/input_ids_e0" # sys.argv[1] #decoded
f2 = "result_data/response_ids_e0" # sys.argv[2] #reference
ref = []
decoded = []

for i, j in zip(sorted(glob.glob(f1+'*.txt')),sorted(glob.glob(f2+'*.txt'))):
        ref_tex = ''
        dec_tex = ''
        for k in open(i).readlines():
                dec_tex = dec_tex + k.strip()
        for l in open(j).readlines():
                ref_tex = ref_tex + l.strip()
        ref.append(ref_tex)
        decoded.append(dec_tex)
print("a")
bl = bleu.moses_multi_bleu(decoded,ref)
print("b")
x = rouge.rouge(decoded,ref)

print(bl)
print(x['rouge_1/f_score'])
print(x['rouge_2/f_score'])
print(x['rouge_l/f_score'])

print("blue: {:.2f}, rouge_1/f_score: {:.2f}, rouge_2/f_score: {:.2f}, rouge_l/f_score: {:.2f}".format(
    bl,x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100))
#print('%.2f\t%.2f\t%.2f\t%.2f'%(bl,x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100))

