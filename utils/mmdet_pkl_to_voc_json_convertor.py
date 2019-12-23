

import pickle
import json
import os 
import sys
import numpy as np

source = sys.argv[1]


# import pdb; pdb.set_trace()
with open(source, 'rb') as f:
    dd = pickle.load(f)



tt = []
for k in range(21):
   tt.append([])


for k in range(21):
	for i in range(4952):
		if k==0:
			tt[k].append([])
		else:
			tt[k].append(dd[i][k-1].tolist())

data = tt

with open(source[:-4]+'.json', 'w') as ff:    
	json.dump(data, ff)
 

print('done!!')
