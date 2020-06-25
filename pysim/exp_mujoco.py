
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import torch
import arcsim
import gc
import time
import json
import sys
import gc
import numpy as np

from datetime import datetime
n_cubes    = 3
n_steps    = 30
goal = torch.tensor([0.0000, 0.0000, 0.0000, 0, 0, 0.2],dtype=torch.float64)

xml_path = './mj_cubes.xml'
model = load_model_from_path(xml_path)
mjsim = MjSim(model)
viewer = MjViewer(mjsim)

ini_state = mjsim.get_state()

now = datetime.now()
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)


with open('conf/rigidcloth/mujoco/mujoco.json','r') as f:
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
save_config(config, out_path+'/conf.json')


torch.set_num_threads(8)
spf = config['frame_steps']
scalev=1

def reset_sim(sim, epoch):
	mjsim.set_state(ini_state)

	arcsim.init_physics(out_path+'/conf.json',out_path+'/out%d'%epoch,False)
	# 0.2 0.65 0.95
	#goal = torch.tensor([0.0000, 0.0000, 0.0000, 0, 0, 0.2],dtype=torch.float64)
	text_name = out_path+'/out%d'%epoch + "/goal.txt"
	np.savetxt(text_name, goal[3:6], delimiter=',')

	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)


def get_loss(ans, param_g):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	#offset1 = torch.tensor([0.0000, 0.0000, 0.0000, 0, 0, 0.45],dtype=torch.float64)
	#offset2 = torch.tensor([0.0000, 0.0000, 0.0000, 0, 0, 0.3],dtype=torch.float64)
	
	sim_state = mjsim.get_state()
	qpos = sim_state[1]
	
	print(qpos)
	#exit()
	#qpos_ideal = [0.95,0,0,0.65,0,0,0.2,0,0]
	#qpos_ideal = [0.2,0,0,-0.1,0,0,-0.55,0,0]
	qpos_ideal = [-0.25,0,0,-0.05,0,0,0,0,0]
	#qpos_ideal = [-0,0,0,-0.25,0,0,-0.05,0,0]
	dis = qpos_ideal-qpos
	dis = dis[::-1]

	goal = ans[2].data.numpy()[3:6] + dis[0:3]
	goal = torch.cat([ans[2].data[0:3], torch.from_numpy(goal)])
	loss2 =  torch.norm(ans[2] - goal, p=2)
	#exit()

	goal = ans[1].data.numpy()[3:6] + dis[3:6]
	goal = torch.cat([ans[1].data[0:3], torch.from_numpy(goal)])
	loss1 = torch.norm(ans[1] - goal, p=2)


	goal = ans[0].data.numpy()[3:6] + dis[6:9]
	goal = torch.cat([ans[0].data[0:3], torch.from_numpy(goal)])
	loss0 = torch.norm(ans[0] - goal, p=2)
	loss  = loss2+loss1+loss0


 
	#print(ans)
	#print(loss)
	#print(param_g)
	return loss

def run_sim(steps,sim,param_g):
	
			
	# sim.obstacles[2].curr_state_mesh.dummy_node.x = param_g[1]
	#print("step")
	#print(dir(mjsim.data))
	#exit()
	for step in range(steps):
		#print(step)

		for i in range(n_cubes):
			dx = [torch.zeros([5],dtype=torch.float64)]
			dx.append(param_g[step][i].unsqueeze(0))
			dx = torch.cat(dx)
			sim.obstacles[i].curr_state_mesh.dummy_node.v += dx

			mjsim.data.qvel[6-3*i]+=param_g[step][i].data

		sim_state = mjsim.get_state()
		
		qvel = sim_state[2]
		

		#print(qvel)
		mjsim.step()
		viewer.render()
		
		'''
		dx = [torch.zeros([5],dtype=torch.float64)]
		dx.append(param_g[step])
		dx = torch.cat(dx,axis=0)
		sim.obstacles[2].curr_state_mesh.dummy_node.v += dx
		'''
		arcsim.sim_step()

	cnt = 0

	ans = []
	for i in range(n_cubes):
		ans.append(sim.obstacles[i].curr_state_mesh.dummy_node.x)
	
	loss = get_loss(ans, param_g)

	return loss, ans

def do_train(cur_step,optimizer,sim,param_g):
	epoch = 0
	while True:
		steps=n_steps
		reset_sim(sim, epoch)
		st = time.time()
		loss, ans = run_sim(steps, sim, param_g)
		en0 = time.time()
		# loss = get_loss(steps,sim)
		optimizer.zero_grad()

		
		# print('step={}'.format(cur_step))
		# print('loss={}'.format(loss.data))
		# f.write('step {}: loss={}\n'.format(cur_step, loss.data))
		# print('step {}: loss={}\n'.format(cur_step, loss.data))

		loss.backward(retain_graph=True)



		# if loss<8e-2:
		# 	break
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		en1 = time.time()
		# print(ans)
		# asdf.asdf
		print("=======================================")
		# print(param_g.data)
		# print(param_g.grad.data)
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

	param_g = torch.zeros([n_steps, 3],dtype=torch.float64, requires_grad=True)

	lr = 0.05
	momentum = 0.4
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.SGD([{'params':param_g,'lr':lr}],momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,param_g)

print("done")

