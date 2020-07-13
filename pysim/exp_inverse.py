import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os
import numpy as np

from datetime import datetime
now = datetime.now()
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

with open('conf/rigidcloth/bounce/bounce.json','r') as f:
	config = json.load(f)


def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')


torch.set_num_threads(8)
spf = config['frame_steps']
scalev=1

def reset_sim(sim, epoch):

	if epoch < 20:

		arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)
		goal = torch.tensor([0.0000, 0.0000, 0.0000, 0, 0, 0],dtype=torch.float64)
		text_name = out_path+'/out%d'%epoch + "/goal.txt"
		np.savetxt(text_name, goal[3:6], delimiter=',')
	else:
		arcsim.init_physics(out_path+'/conf.json',out_path+'/out',False)

	# arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out',False)

	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)


def get_loss(ans, param_g):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	vec = torch.tensor([0, 0],dtype=torch.float64)
	loss = torch.norm(ans.narrow(0, 3, 2) - vec, p=2)
	reg  = torch.norm(param_g, p=2)*0.001
 
	print(ans)
	print(loss)
	print(reg)

	return loss + reg

def run_sim(steps,sim,param_g):
	
			
	for obstacle in sim.obstacles:
		for node in obstacle.curr_state_mesh.nodes:
			node.m    *= 0.01

	# sim.obstacles[2].curr_state_mesh.dummy_node.x = param_g[1]
	print("step")
	for step in range(20):
		print(step)

		dx = []
		dx.append(torch.zeros([3],dtype=torch.float64))
		dx.append(param_g[step])
		dx.append(torch.zeros([1],dtype=torch.float64))
		dx = torch.cat(dx)
		sim.obstacles[0].curr_state_mesh.dummy_node.v += dx

		arcsim.sim_step()

	cnt = 0

	ans = sim.obstacles[0].curr_state_mesh.dummy_node.x
	ans = ans

	# ans = torch.tensor([0, 0, 0],dtype=torch.float64)
	# for node in sim.cloths[0].mesh.nodes:
	# 	cnt += 1
	# 	ans = ans + node.x
	# ans /= cnt
	loss = get_loss(ans, param_g)

	return loss, ans

def do_train(cur_step,optimizer,sim,param_g):
	epoch = 0
	while True:
		steps=10
		reset_sim(sim, epoch)
		st = time.time()
		loss, ans = run_sim(steps, sim, param_g)
		en0 = time.time()
		optimizer.zero_grad()

	
		loss.backward(retain_graph=True)



		# if loss<8e-2:
		# 	break
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		en1 = time.time()
		# print(ans)
		# asdf.asdf
		print("=======================================")
		#print(param_g.data)
		#print(param_g.grad.data)
		f.write('epoch {}:  loss={} \n'.format(epoch,  loss.data))
		print('epoch {}:  loss={} \n'.format(epoch, loss.data))

		print('forward time={}'.format(en0-st))
		print('backward time={}'.format(en1-en0))


		optimizer.step()
		if epoch>=100:
			quit()
		epoch = epoch + 1
		# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	# reset_sim(sim)

	param_g = torch.zeros([20, 2],dtype=torch.float64, requires_grad=True)

	lr = 0.1
	momentum = 0.4
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.SGD([{'params':param_g,'lr':lr}],momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,param_g)

print("done")

