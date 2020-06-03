import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os

print(sys.argv)#prefix
if not os.path.exists(sys.argv[1]):
	os.mkdir(sys.argv[1])

with open('conf/rigidcloth/q_rigid_gravity.json','r') as f:
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

def get_loss(ans):
	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
	vec = torch.tensor([0.0000, 0.0000, 0.0000, 0.7, 0.6, -3],dtype=torch.float64)
	loss = torch.norm(ans - vec, p=2)

	return loss

def run_sim(steps,sim,param_g):
	sim.obstacles[0].curr_state_mesh.dummy_node.v = param_g
	for step in range(50):
		print(step)
		arcsim.sim_step()

	ans  = sim.obstacles[0].curr_state_mesh.dummy_node.x
	loss = get_loss(ans)

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



		if loss<1e-3:
			break
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		en1 = time.time()


		print("=======================================")
		f.write('epoch {}: g={} loss={} grad={}\n ans={}\n'.format(epoch, param_g.data, loss.data, param_g.grad.data, ans.data))
		print('epoch {}: g={} loss={} grad={}\n ans={}\n'.format(epoch, param_g.data, loss.data, param_g.grad.data, ans.data))

		print('forward time={}'.format(en0-st))
		print('backward time={}'.format(en1-en0))


		optimizer.step()
		epoch = epoch + 1
		# break

with open(sys.argv[1]+'/log.txt','w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	reset_sim(sim)

	param_g = torch.tensor([0,0,0,0,0,0],dtype=torch.float64, requires_grad=True)

	lr = 0.1
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.SGD([{'params':param_g,'lr':lr}],momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,param_g)

print("done")

