import torch
import arcsim
import json
import sys
import gc
import os
import copy
import numpy as np
import time

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

def reset_sim():
	with torch.no_grad():
		arcsim.init_physics('conf/rigidcloth/circular_domino/circular_domino_make.json',out_path+'/out',False)

threshold = 70/360*2*3.1415926

def run_sim(sim):
	with torch.no_grad():
		for step in range(400):

			if step == 70:
				for obstacle in sim.obstacles:
					for node in obstacle.curr_state_mesh.nodes:
						node.m    *= 80

			for obstacle in sim.obstacles:
				x = obstacle.curr_state_mesh.dummy_node.x.data
				x = np.abs(x)
				# if  x[0]>threshold or x[1]>threshold:
				if  x[5]<0.4:
					obstacle.curr_state_mesh.dummy_node.v = torch.cat(
						[ obstacle.curr_state_mesh.dummy_node.v.narrow(0, 0, 1),
						obstacle.curr_state_mesh.dummy_node.v.narrow(0, 1, 1)*0.5,
						obstacle.curr_state_mesh.dummy_node.v.narrow(0, 2, 1),
						obstacle.curr_state_mesh.dummy_node.v.narrow(0, 3, 1)*0.5,
						obstacle.curr_state_mesh.dummy_node.v.narrow(0, 4, 1)*0.5,
						obstacle.curr_state_mesh.dummy_node.v.narrow(0, 5, 1)
						]
						)

			if step < 30:
				for node in sim.cloths[0].mesh.nodes:
					node.v    += torch.tensor([2, 0, 0],dtype=torch.float64)
			# else:
			# 	inc         = np.maximum(np.exp(-(step-30.0)/50.0),0.2)
			# 	inc			= inc * np.cos((step-30)/40) *torch.tensor([2, 0, 0],dtype=torch.float64)
			# 	for node in sim.cloths[0].mesh.nodes:
			# 		node.v    += inc
			# 	print(inc)


			arcsim.sim_step()
			print("step:")
			print(step)

#def make_json (num_do=2, radius=3, total_angle=50):
def make_json (num_do=15, radius=3, total_angle=360):
	with open('conf/rigidcloth/circular_domino/circular_domino2.json','r') as f:
		config = json.load(f)

	cube = config['obstacles'][0]


	def save_config(config, file):
		with open(file,'w') as f:
			json.dump(config, f)

	angle = total_angle / (num_do)
	last_do = num_do 

	for i in range(1, last_do):
		new_cube = copy.deepcopy(cube)
		this_angle = i*angle


		new_cube['transform']['translate'][0] = radius * np.sin(this_angle / 180 * np.pi) 
		new_cube['transform']['translate'][1] = radius * np.cos(this_angle / 180 * np.pi)
		new_cube['transform']['rotate'][0]    = -this_angle
		# if i == last_do - 1:
		# 	new_cube['transform']['scale'] = 1.25
		# 	# new_cube['transform']['translate'][2] += 0.2 
		# 	new_cube['transform']['translate'][1] -= 0.5
		# 	new_cube['transform']['translate'][0] -=  0
		config['obstacles'].append(new_cube)

	this_angle = 0
	cube['transform']['translate'][0] = radius * np.sin(this_angle / 180 * np.pi) + 0.35
	cube['transform']['translate'][1] = radius * np.cos(this_angle / 180 * np.pi)
	cube['transform']['rotate'][0]    = -this_angle


	flag = config['cloths'][0]
	flag['transform']['translate'][0] += radius * np.sin(this_angle / 180 * np.pi)
	flag['transform']['translate'][1] += radius * np.cos(this_angle / 180 * np.pi)
	flag['transform']['translate'][0] -= 0.4
	flag['transform']['translate'][2] = 1.6
	flag['transform']['scale'] = 2
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


# arcsim.msim(4,['arcsim','simulateoffline','conf/rigidcloth/multibody/multibody_make.json','out'])
# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/circular_domino/circular_domino_make.json','200122_circulat_dominod'])
reset_sim()
sim = arcsim.get_sim()

run_sim(sim)

# time_record = time.time() - t
# # time_per    = time_record / (max_line*max_line)



# print(time_record)
# print(time_per)

# f = open("circular_domino.txt", "a")
# f.write("%d %f %f\n"%(max_line, time_record, time_per))