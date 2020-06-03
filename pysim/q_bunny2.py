import torch
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import sys
import gc
import numpy as np
import os

print(sys.argv)#prefix
if not os.path.exists(sys.argv[1]):
	os.mkdir(sys.argv[1])


handles = [30,25,60,54]
# handles = [54]
target = np.array([0, 0, 0.5])


with open('conf/rigidcloth/clothbunny/clothbunny2.json','r') as f:
	config = json.load(f)
# matfile = config['cloths'][0]['materials'][0]['data']
# with open(matfile,'r') as f:
# 	matconfig = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

# save_config(matconfig, sys.argv[1]+'/mat.json')
# save_config(matconfig, sys.argv[1]+'/orimat.json')
# config['cloths'][0]['materials'][0]['data'] = sys.argv[1]+'/mat.json'
# config['end_time']=20
save_config(config, sys.argv[1]+'/conf.json')


torch.set_num_threads(8)
spf = config['frame_steps']

scalev=1

def reset_sim(sim):
	arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out2',False)
	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)


def get_loss(ans, goal):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	dif = ans - goal
	loss = torch.norm(dif.narrow(0, 3, 3), p=2)

	return loss

def run_sim(sim):
	for obstacle in sim.obstacles:
		for node in obstacle.curr_state_mesh.nodes:
			node.m    *= 0.002

	for step in range(200):

		target = np.array([0, 0, 0.5])

		if (step> 50):
			target += np.array([0, 0, 0.3])

		for i in range(len(handles)):
		
			this_x = sim.cloths[0].mesh.nodes[handles[i]].x.data
			this_x = np.array(this_x)
			print(this_x)
			vec = target - this_x
			vec = vec / np.linalg.norm(vec)
			print(vec)
			# sim.cloths[0].mesh.nodes[handles[i]].x +=  torch.tensor([0,0,0.04])
			sim.cloths[0].mesh.nodes[handles[i]].v =  torch.tensor(vec*2, dtype=torch.float64)
			# sim.cloths[0].mesh.nodes[handles[i]].v +=  torch.tensor([0,0,0.04])
		# for i in range(len(sim.cloths[0].mesh.nodes)):
		# 	print(target)
		# 	this_x = sim.cloths[0].mesh.nodes[i].x.data
		# 	this_x = np.array(this_x)
		# 	print(this_x)
		# 	vec = target - this_x
		# 	vec = vec / np.linalg.norm(vec)
		# 	print(vec)
		# 	sim.cloths[0].mesh.nodes[i].v =  torch.tensor(vec*3)
		# 	#sim.cloths[0].mesh.nodes[i].x =  sim.cloths[0].mesh.nodes[i].x+torch.tensor(vec*0.03)

		arcsim.sim_step()



with open(sys.argv[1]+'/log.txt','w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	reset_sim(sim)
	run_sim(sim)
	#param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)


	#optimizer = torch.optim.SGD([{'params':net.parameters(),'lr':lr}],momentum=momentum)


print("done")

