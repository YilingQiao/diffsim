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


with open('conf/rigidcloth/momentum.json','r') as f:
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

	arcsim.init_physics(out_path+'/conf.json',out_path+'/out%d'%epoch,False)
	goal = torch.tensor([0.0000, 0.0000, 0.0000, 0, 0, 0],dtype=torch.float64)
	text_name = out_path+'/out%d'%epoch + "/goal.txt"
	np.savetxt(text_name, goal[3:6], delimiter=',')


	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)


def get_loss(ans, param_g):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	vec = torch.tensor([3],dtype=torch.float64)
	total_p = ans[0] + param_g*ans[1]
	print(ans)
	print(total_p)	
	loss = torch.norm(total_p - vec, p=2)
	#reg  = torch.norm(param_g, p=2)*0.05
 
	#print(ans)
	print("-----")
	print(loss)
	print(param_g)
	#print(reg)

	return loss #+ reg

def run_sim(steps,sim,param_g):
	
	
	for node in sim.obstacles[1].curr_state_mesh.nodes:
		node.m    *= param_g


	# sim.obstacles[2].curr_state_mesh.dummy_node.x = param_g[1]
	#print("step")
	for step in range(20):
		#print(step)

		
		arcsim.sim_step()

	cnt = 0
	ans = []
	ans.append(sim.obstacles[0].curr_state_mesh.dummy_node.v[3])
	ans.append(sim.obstacles[1].curr_state_mesh.dummy_node.v[3])

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
		steps=4*25*spf
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
		print(param_g.grad.data)
		f.write('epoch {}:  loss={} \n'.format(epoch,  loss.data))
		print('epoch {}:  loss={} \n'.format(epoch, loss.data))

		print('forward time={}'.format(en0-st))
		print('backward time={}'.format(en1-en0))


		optimizer.step()
		if epoch>=300:
			quit()
		epoch = epoch + 1
		# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	# reset_sim(sim)

	param_g = torch.tensor(1.5,dtype=torch.float64, requires_grad=True)

	lr = 0.2
	momentum = 0.01
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.SGD([{'params':param_g,'lr':lr}],momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,param_g)

print("done")

