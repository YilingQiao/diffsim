import torch
import arcsim
import gc
import time
import json

with open('conf/rigidcloth/q_rigid_gravity.json','r') as f:
	config = json.load(f)
# matfile = config['cloths'][0]['materials'][0]['data']
# with open(matfile,'r') as f:
# 	matconfig = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

# save_config(matconfig, 'curmat.json')
# config['cloths'][0]['materials'][0]['data'] = 'curmat.json'
# save_config(config, 'curconf.json')


torch.set_num_threads(8)
sim=arcsim.get_sim()

def reset_sim():
	arcsim.init_physics('conf/rigidcloth/q_rigid_gravity.json','',False)
	g = sim.gravity
	g.requires_grad = True
	return g

def run_sim():
	for step in range(49):
		arcsim.sim_step()

def get_loss():

	print(sim.obstacles[0].curr_state_mesh.dummy_node.x)

	vec = torch.tensor([0,0,0,0,0,0],dtype=torch.float64)


	ans = torch.norm(sim.obstacles[0].curr_state_mesh.dummy_node.x - vec, p=2)

	print (ans)

	return ans 


lr = 0.01
momentum = 0.7

with open('log.txt','w',buffering=1) as f:
	tot_step = 20
	for cur_step in range(tot_step):
		g = reset_sim()
		st = time.time()
		run_sim()
		print("=======================================")
		en0 = time.time()
		loss = get_loss()
		print('step={}'.format(cur_step))
		print('forward time={}'.format(en0-st))
		loss.backward(retain_graph=True)
		en1 = time.time()
		print('backward time={}'.format(en1-en0))
		f.write('step {}: g={} loss={} grad={}\n'.format(cur_step, g.data, loss.data, g.grad.data))
		print('step {}: g={} loss={} grad={}\n'.format(cur_step, g.data, loss.data, g.grad.data))
		g.data -= lr * g.grad
		print(g.data)
		#if g.norm()>9.8:
		#	g.data *=9.8/g.norm()
		config['gravity'] = g.detach().numpy().tolist()
		save_config(config, 'conf/rigidcloth/q_rigid_gravity.json')


