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

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight%s.pth'%timestamp)

class Net(nn.Module):
	def __init__(self, n_input, n_output):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(n_input, 400).double()
		self.fc2 = nn.Linear(400, 300).double()
		self.fc3 = nn.Linear(300, n_output).double()
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

with open('conf/rigidcloth/manipulation/manipulation.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')


torch.set_num_threads(8)
spf = config['frame_steps']

scalev=1

def reset_sim(sim):
	arcsim.init_physics(out_path+'/conf.json', out_path+'/out',False)
	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)


def get_loss(ans, goal):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	dif = ans - goal
	loss = torch.norm(dif.narrow(0, 3, 3), p=2)

	return loss

def run_sim(steps, sim, net, goal):


	#sim.obstacles[0].curr_state_mesh.dummy_node.x = torch.tensor([0.0000, 0.0000, 0.0000,
	#np.random.random(), np.random.random(), -np.random.random()],dtype=torch.float64)

	for step in range(steps):
		remain_time = torch.tensor([(steps - step)/steps],dtype=torch.float64)
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
		steps = int(1*20*spf)
		reset_sim(sim)

		sigma = 0.1
		x = np.random.random()*sigma - 0.5*sigma + np.random.randint(2)*2-1

		goal = torch.tensor([0.0000, 0.0000, 0.0000,
		x, 
		0, 
		2+np.random.random()*sigma - 0.5*sigma],dtype=torch.float64)
		# goal = torch.tensor([0.0000, 0.0000, 0.0000,
		# x, 
		# 0, 
		# 2+np.random.random()*sigma - 0.5*sigma],dtype=torch.float64)

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
		if epoch>=200:
			quit()
		# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	reset_sim(sim)

	#param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
	net = Net(13, 3)
	if os.path.exists(torch_model_path):
		net.load_state_dict(torch.load(torch_model_path))
		print("load: %s\n success" % torch_model_path)

	lr = 0.001
	momentum = 0.6
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	#optimizer = torch.optim.SGD([{'params':net.parameters(),'lr':lr}],momentum=momentum)
	optimizer = torch.optim.Adam(net.parameters(),lr=lr)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,net)

print("done")

