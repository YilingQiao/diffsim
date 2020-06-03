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
	arcsim.init_physics('conf/rigidcloth/circular_domino/circular_domino_make.json','',False)


def run_sim():
	for step in range(300):
		arcsim.sim_step()
		print(step)

def make_json (num_do=25, radius=4, total_angle=360):
	with open('conf/rigidcloth/circular_domino/circular_domino.json','r') as f:
		config = json.load(f)

	cube = config['obstacles'][0]

	def save_config(config, file):
		with open(file,'w') as f:
			json.dump(config, f)

	angle = total_angle / (num_do)
	last_do = num_do - 1 

	for i in range(1, last_do):
		new_cube = copy.deepcopy(cube)
		this_angle = i*angle
		print(this_angle)
		print( np.cos(this_angle / 180 * np.pi))


		new_cube['transform']['translate'][0] = radius * np.sin(this_angle / 180 * np.pi) 
		new_cube['transform']['translate'][1] = radius * np.cos(this_angle / 180 * np.pi)
		new_cube['transform']['rotate'][0]    = -this_angle
		if i == last_do - 1:
			new_cube['transform']['scale'] = 1.25
			# new_cube['transform']['translate'][2] += 0.2 
			new_cube['transform']['translate'][1] -= 0.5
			new_cube['transform']['translate'][0] -=  0
		config['obstacles'].append(new_cube)

	this_angle = 0
	cube['transform']['translate'][0] = radius * np.sin(this_angle / 180 * np.pi) + 0.35
	cube['transform']['translate'][1] = radius * np.cos(this_angle / 180 * np.pi)
	cube['transform']['rotate'][0]    = -this_angle


	config['cloths'][0]['transform']['translate'][0] += radius * np.sin(this_angle / 180 * np.pi)
	config['cloths'][0]['transform']['translate'][1] += radius * np.cos(this_angle / 180 * np.pi)

	# small_cube = config['obstacles'][2]

	# small_cube['transform']['translate'][0] = radius * np.sin(this_angle / 180 * np.pi)
	# small_cube['transform']['translate'][1] = radius * np.cos(this_angle / 180 * np.pi)
	# small_cube['transform']['translate'][0] -= 0.2


	save_config(config, "conf/rigidcloth/circular_domino/circular_domino_make.json")

print(sys.argv)

# max_line = int(sys.argv[1])

# print("max_line")
# print(max_line)
# make_json(max_line)
make_json()


#arcsim.msim(4,['arcsim','simulateoffline','conf/rigidcloth/multibody/multibody_make.json','out'])
arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/circular_domino/circular_domino_make.json','200122_circulat_dominod'])
# reset_sim()
# t = time.time()
# run_sim()

# time_record = time.time() - t
# # time_per    = time_record / (max_line*max_line)



# print(time_record)
# print(time_per)

# f = open("circular_domino.txt", "a")
# f.write("%d %f %f\n"%(max_line, time_record, time_per))