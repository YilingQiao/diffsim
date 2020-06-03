import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os

from datetime import datetime
now = datetime.now()
timestamp = datetime.timestamp(now)


print(sys.argv)#prefix
if not os.path.exists(sys.argv[1]):
	os.mkdir(sys.argv[1])

handles = [0,1,2,3]

with open('conf/rigidcloth/bounce/bounce.json','r') as f:
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
	arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out',False)

	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)


def get_loss(ans, param_g):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	vec = torch.tensor([0, 0],dtype=torch.float64)
	loss = torch.norm(ans.narrow(0, 3, 2) - vec, p=2)
	reg  = torch.norm(param_g, p=2)*0.05

	print(ans)
	print(loss)
	print(reg)

	return loss + reg

def run_sim(steps,sim,param_g):
	
	# sim.obstacles[2].curr_state_mesh.dummy_node.x = param_g[1]
	for step in range(20):
		print("step")
		print(step)

		dx = []
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
		steps=4*25*spf
		reset_sim(sim)
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

with open(sys.argv[1]+('/log%f.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	reset_sim(sim)

	param_g = torch.zeros([20, 5],dtype=torch.float64, requires_grad=True)

	lr = 0.03
	momentum = 0.4
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.SGD([{'params':param_g,'lr':lr}],momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,param_g)

print("done")

