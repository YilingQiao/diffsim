import torch
import arcsim
import json
import sys
import gc
import os
import copy
import numpy as np
import time

def reset_sim():
	arcsim.init_physics('conf/rigidcloth/multibody/multi_make.json','',False)


def run_sim():
	for step in range(20):
		arcsim.sim_step()
		print(step)

perline = 20

def make_json (max_block = 1):
	with open('conf/rigidcloth/multi/multi.json','r') as f:
		config = json.load(f)


	single_length = 0.3

	ini_size      = single_length
	ini_x         = 0
	

	def save_config(config, file):
		with open(file,'w') as f:
			json.dump(config, f)

	for i in range(max_block):

		for j in range(4):
			prim = config['obstacles'][j]
			for k in range(1, 5):
				new_prim = copy.deepcopy(prim)
				new_prim['transform']['translate'][0] = ini_x + k * single_length
				# new_prim['transform']['rotate'][0] = np.random.random() * 360
				config['obstacles'].append(new_prim)
			prim['transform']['translate'][0] = ini_x

		ini_x += 6*single_length



	save_config(config, "conf/rigidcloth/multi/multi_make.json")

print(sys.argv)

max_block = int(sys.argv[1])

print("max_block")
print(max_block)
make_json(max_block)


#arcsim.msim(4,['arcsim','simulateoffline','conf/rigidcloth/multibody/multibody_make.json','out'])
arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/multi/multi_make.json','out'])
reset_sim()
t = time.time()
# run_sim()

time_record = time.time() - t
time_per    = time_record / (max_block*perline)



print(time_record)
print(time_per)

f = open("multi.txt", "a")
f.write("%d %f %f\n"%(max_block, time_record, time_per))