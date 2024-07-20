## load head score file, llama-2-7b-80k for example

import os
import json
import numpy as np

head_score_file = './head_score/llama-2-7b-80k.json'

with open(head_score_file) as file:
    head_list = json.loads(file.readline())
## use the average retrieval score and ranking
head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True) 
top_retrieval_heads = [[l[0],  round(np.mean(l[1]), 2)] for l in head_score_list][:10]
print(top_retrieval_heads)


# save top heads

model_name = head_score_file.split("/")[-1].split(".")[0]
os.makedirs('./top_heads', exist_ok=True)
with open(f'./top_heads/{model_name}_top_heads.json', 'w') as file:
	json.dump(top_retrieval_heads, file)