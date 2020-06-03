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
from datetime import datetime

print(sys.argv)#prefix
if not os.path.exists(sys.argv[1]):
	os.mkdir(sys.argv[1])

now = datetime.now()
timestamp = datetime.timestamp(now)

torch_model_path = sys.argv[1] + ('/net_weight.pth%f'%timestamp)

class Net(nn.Module):
	def __init__(self, n_input, n_output):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(n_input, 50).double()
		self.fc2 = nn.Linear(50, 200).double()
		self.fc3 = nn.Linear(200, n_output).double()
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

with open('conf/rigidcloth/manipulation/manipulation_vid.json','r') as f:
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

def reset_sim(sim, epoch, goal):
	if epoch % 4==0:
		arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out%d'%epoch,False)
		text_name = sys.argv[1]+'/out%d'%epoch + "/goal.txt"
	
		np.savetxt(text_name, goal[3:6], delimiter=',')
	else:
		arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out',False)
	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)

	for i in range(10, len(sim.obstacles[0].curr_state_mesh.nodes)):
		this_node   = sim.obstacles[0].curr_state_mesh.nodes[i]
		this_node.m = this_node.m*0.00001 
		


def get_loss(ans, goal):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	dif = ans - goal
	loss = torch.norm(dif.narrow(0, 3, 3), p=2)

	return loss

def run_sim(steps, sim, net, goal):


	#sim.obstacles[0].curr_state_mesh.dummy_node.x = torch.tensor([0.0000, 0.0000, 0.0000,
	#np.random.random(), np.random.random(), -np.random.random()],dtype=torch.float64)

	for step in range(steps):
		remain_time = torch.tensor([(steps - step)/50],dtype=torch.float64)
		net_output = net(torch.cat([sim.obstacles[0].curr_state_mesh.dummy_node.x - goal,
									sim.obstacles[0].curr_state_mesh.dummy_node.v, 
									remain_time]))
		sim_input = torch.cat([torch.zeros([3], dtype=torch.float64), net_output])
		sim.obstacles[1].curr_state_mesh.dummy_node.v += sim_input 
	
		arcsim.sim_step()


	ans  = sim.obstacles[0].curr_state_mesh.dummy_node.x
	loss = get_loss(ans, goal)


	return loss, ans

def do_train(cur_step,optimizer,sim,net):
	epoch = 0
	while True:
		steps = int(1*15*spf)

		sigma = 0.4
		x = np.random.random()*sigma - 0.5*sigma + np.random.randint(2)*2-1
		ini_co = torch.tensor([0,  0, 0,  4.5549e-04, -2.6878e-01, 0.23], dtype=torch.float64)
		goal = ini_co + torch.tensor([0.0000, 0.0000, 0.0000,
		x, 
		0, 
		2+np.random.random()*sigma - 0.5*sigma],dtype=torch.float64)
		# goal = torch.tensor([0.0000, 0.0000, 0.0000,
		# x, 
		# 0, 
		# 2+np.random.random()*sigma - 0.5*sigma],dtype=torch.float64)

		reset_sim(sim, epoch, goal)

		st = time.time()
		loss, ans = run_sim(steps, sim, net, goal)
		en0 = time.time()
		
		optimizer.zero_grad()

		
		# print('step={}'.format(cur_step))
		# print('loss={}'.format(loss.data))
		# f.write('step {}: loss={}\n'.format(cur_step, loss.data))
		# print('step {}: loss={}\n'.format(cur_step, loss.data))

		loss.backward()

		en1 = time.time()
		print("=======================================")
		f.write('epoch {}: loss={}\n  ans = {}\n goal = {}\n'.format(epoch, loss.data,  ans.data, goal.data))
		print('epoch {}: loss={}\n  ans = {}\n goal = {}\n'.format(epoch, loss.data,  ans.data, goal.data))
		#print('epoch {}: loss={}  ans={}\n'.format(epoch, loss.data, ans.data))

		print('forward tim = {}'.format(en0-st))
		print('backward time = {}'.format(en1-en0))


		if epoch % 5 == 0:
			torch.save(net.state_dict(), torch_model_path)

		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])





		optimizer.step()
		epoch = epoch + 1
		if epoch>=60:
			quit()
		# break

with open(sys.argv[1]+('/log%f.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	# reset_sim(sim)

	#param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
	net = Net(13, 3)
	if os.path.exists(torch_model_path):
		net.load_state_dict(torch.load(torch_model_path))
		print("load: %s\n success" % torch_model_path)

	lr = 0.001
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	#optimizer = torch.optim.SGD([{'params':net.parameters(),'lr':lr}],momentum=momentum)
	optimizer = torch.optim.Adam(net.parameters(),lr=lr)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,net)

print("done")

