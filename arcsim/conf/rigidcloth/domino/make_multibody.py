import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os
import copy



with open('multibody.json','r') as f:
	config = json.load(f)

cube = config['obstacles'][0]

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

num_cube = 20

# for i in range(num_cube):
# 	new_cube = copy.deepcopy(cube)
# 	new_cube['transform']['translate'][1] -= 0.03 * (i+1)
# 	config['obstacles'].append(new_cube)
# 	print(config)
for i in range(num_cube):
	new_cube = copy.deepcopy(cube)
	new_cube['transform']['translate'][1] -= 0.03 * (i+1)
	config['obstacles'].append(new_cube)
	print(config)

save_config(config, "multibody_make.json".format(num_cube))

