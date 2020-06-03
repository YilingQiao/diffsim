import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os

print(sys.argv)#prefix stdout mat_name
if not os.path.exists(sys.argv[1]):
	os.mkdir(sys.argv[1])

# with open('conf/demo_collision2.json','r') as f:
with open('conf/demo_collision.json','r') as f:
	config = json.load(f)
matfile = config['cloths'][0]['materials'][0]['data']
with open(matfile,'r') as f:
	matconfig = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(matconfig, sys.argv[1]+'/mat.json')
save_config(matconfig, sys.argv[1]+'/orimat.json')
config['cloths'][0]['materials'][0]['data'] = sys.argv[1]+'/mat.json'
config['end_time']=20
save_config(config, sys.argv[1]+'/conf.json')


torch.set_num_threads(8)

scaleden = 1
scalestr = 1e3
scaleben = 1e-4

def reset_sim(sim):
	arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out',False)
	mat = sim.cloths[0].materials[0]
	density = mat.densityori
	stretch = mat.stretchingori
	bend = mat.bendingori
	return density, stretch, bend

def get_loss(steps,sim):
	diffs = []
	for node in sim.cloths[0].mesh.nodes:
		diffs.append(node.x.norm())
	return torch.stack(diffs).mean()

def run_sim(steps,sim):
	losses=[]
	for step in range(steps):
		arcsim.sim_step()
	losses.append(get_loss(1,sim))
	return torch.stack(losses).mean()

def do_train(cur_step,optimizer,sim):
	while True:
		steps=1#min(50,cur_step*2+2)
		density,stretch,bend = reset_sim(sim)
		st = time.time()
		loss = run_sim(steps, sim)
		en0 = time.time()
		# loss = get_loss(steps,sim)
		optimizer.zero_grad()
		print('step={}'.format(cur_step))
		print('forward time={}'.format(en0-st))
		f.write('forward time={}\n'.format(en0-st))
		print('loss={}'.format(loss.data))
		f.write('step {}: d={} loss={}\n'.format(cur_step, density.data*scaleden, loss.data))
		print('step {}: d={} loss={}\n'.format(cur_step, density.data*scaleden, loss.data))
		loss.backward()
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		en1 = time.time()
		print('backward time={}'.format(en1-en0))
		f.write('backward time={}\n'.format(en1-en0))
		print(density.grad)
		print(stretch.grad)
		print(bend.grad)
		break

with open(sys.argv[1]+'/log.txt','w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	density,stretch,bend = reset_sim(sim)
	config['cloths'][0]['materials'][0]['reuse']=True
	save_config(config, sys.argv[1]+'/conf.json')
	lr = 0.1
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.SGD([{'params':density,'lr':lr}, {'params':stretch,'lr':lr}, {'params':bend,'lr': lr*100}],momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim)

print("done")

