

import pickle
import json
import os 
import sys
import numpy as np

source = sys.argv[1]


# import pdb; pdb.set_trace()
with open(source, 'rb') as f:
    dd = json.load(f)



tt = []

for k in range(21):
	tt.append([])	
	for i in range(4952):
			tt[k].append([])




img_ods = dict()
count = 0
for item in dd:
	if item['image_id'] not in img_ods:
		img_ods[item['image_id']] = count
		count += 1


for item in dd:
	box = item['bbox']
	box[2] += box[0]
	box[3] += box[1]
	box.append(item['score'])
	aa = img_ods[item['image_id']]
	print(item['category_id'], ' ', aa)
	tt[item['category_id']][aa].append(box)


data = tt

with open(source[:-5]+'_converted.json', 'w') as ff:    
	json.dump(data, ff)
 

print('done!!')
