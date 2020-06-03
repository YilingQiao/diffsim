import torch
import arcsim
import gc
import time
import json

with open('conf/gravity.json','r') as f:
	config = json.load(f)
matfile = config['cloths'][0]['materials'][0]['data']
with open(matfile,'r') as f:
	matconfig = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(matconfig, 'curmat.json')
config['cloths'][0]['materials'][0]['data'] = 'curmat.json'
save_config(config, 'curconf.json')


torch.set_num_threads(8)
sim=arcsim.get_sim()

def reset_sim():
	arcsim.init_physics('curconf.json','',False)
	g = sim.gravity
	g.requires_grad = True
	return g

def run_sim():
	for step in range(49):
		arcsim.sim_step()

def get_loss():
	vec = torch.tensor([0,0,-1],dtype=torch.float64)
	ans = torch.zeros([],dtype=torch.float64)
	cnt = 0
	for node in sim.cloths[0].mesh.nodes:
		cnt += 1
		ans = ans + torch.dot(node.x,vec)
	return ans / cnt

lr = 10

with open('log.txt','w',buffering=1) as f:
	tot_step = 2
	for cur_step in range(tot_step):
		g = reset_sim()
		st = time.time()
		run_sim()
		en0 = time.time()
		loss = get_loss()
		print('step={}'.format(cur_step))
		print('forward time={}'.format(en0-st))
		loss.backward()
		en1 = time.time()
		print('backward time={}'.format(en1-en0))
		f.write('step {}: g={} loss={} grad={}\n'.format(cur_step, g.data, loss.data, g.grad.data))
		print('step {}: g={} loss={} grad={}\n'.format(cur_step, g.data, loss.data, g.grad.data))
		g.data -= lr * g.grad
		#if g.norm()>9.8:
		#	g.data *=9.8/g.norm()
		config['gravity'] = g.detach().numpy().tolist()
		save_config(config, 'curconf.json')


