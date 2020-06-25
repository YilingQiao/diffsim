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
	arcsim.init_physics('conf/rigidcloth/multibody/multibody_make.json','',False)


def run_sim():
	for step in range(20):
		arcsim.sim_step()
		print(step)

perline = 50

def make_json (max_line = 2):
	with open('conf/rigidcloth/multibody/multibody.json','r') as f:
		config = json.load(f)

	cube   = config['obstacles'][0]
	ground = config['obstacles'][1]


	single_length = 0.04

	ini_size      = single_length
	ini_x         = -4
	ini_y         = -4

	def save_config(config, file):
		with open(file,'w') as f:
			json.dump(config, f)

	cube['transform']['scale'] = 2**(0)
	cube['transform']['translate'][1] = ini_y 
	cube['transform']['translate'][0] = ini_x 
	cube['transform']['translate'][2] = 0.2

	for i in range(0, max_line):
		ground['transform']['scale'] = 2**(i)
		for j in range(perline):
			if i == 0 and j == 0:
				continue
			new_cube = copy.deepcopy(cube)
			new_cube['transform']['scale'] = ini_size/single_length
			new_cube['transform']['translate'][1] = ini_y + j*ini_size
			new_cube['transform']['translate'][0] = ini_x 
			new_cube['transform']['translate'][2] = 0.2
			config['obstacles'].append(new_cube)
			#print(config)
	
		ini_size = single_length * (i+1) * 4
		ini_x    += ini_size

	save_config(config, "conf/rigidcloth/multibody/multibody_make.json")

print(sys.argv)

max_line = int(sys.argv[1])

print("max_line")
print(max_line)
make_json(max_line)


#arcsim.msim(4,['arcsim','simulateoffline','conf/rigidcloth/multibody/multibody_make.json','out'])
#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/multibody/multibody_make.json','out'])
reset_sim()
t = time.time()
run_sim()

time_record = time.time() - t
time_per    = time_record / (max_line*perline)



print(time_record)
print(time_per)

f = open("multibody.txt", "a")
f.write("%d %f %f\n"%(max_line, time_record, time_per))