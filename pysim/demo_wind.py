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

with open('conf/demo_wind.json','r') as f:
	config = json.load(f)
matfile = config['cloths'][0]['materials'][0]['data']
with open(matfile,'r') as f:
	matconfig = json.load(f)
with open('materials/'+sys.argv[3],'r') as f:
	matstd = json.load(f)

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
	fstd = sys.argv[2]+'/%04d_00.obj'%(steps/2)
	mesh = arcsim.Mesh()
	arcsim.load_obj(mesh, fstd)
	diffs = []
	for node0,node1 in zip(mesh.nodes, sim.cloths[0].mesh.nodes):
		diffs.append((node0.x-node1.x)*(node0.x-node1.x))
	arcsim.delete_mesh(mesh)
	return torch.stack(diffs).sum(dim=1).mean()

def run_sim(steps,sim):
	losses=[]
	for step in range(steps):
		arcsim.sim_step()
		if step % 2 == 1:
			losses.append(get_loss(step+1,sim))
	return torch.stack(losses).mean()

def get_dis(density, stretch, bend):
	with torch.no_grad():
		f.write('grads:\n')
		f.write('{}\n'.format(density*scaleden))
		f.write('{}\n'.format(stretch*scalestr))
		f.write('{}\n'.format(bend*scaleben))
		ld = (torch.tensor(matstd['density'],dtype=torch.float64)-density*scaleden).norm().data
		ls = (torch.tensor(matstd['stretching'],dtype=torch.float64)-stretch*scalestr).norm().data
		lb = (torch.tensor(matstd['bending'],dtype=torch.float64)-bend*scaleben).norm().data
		print('dis={} {} {}\n'.format(ld,ls,lb))
		f.write('dis={} {} {}\n'.format(ld,ls,lb))

def renew_loss():
	print('renew',steps)
	matconfig['density'] = density.detach().numpy().tolist()
	matconfig['stretching'] = stretch.detach().numpy().tolist()
	matconfig['bending'] = bend.detach().numpy().tolist()
	save_config(matconfig, sys.argv[1]+'/mat.json')
	reset_sim(sim)
	loss = run_sim(steps, sim)
	loss.backward()
	return loss

def do_train(cur_step,optimizer,sim):
	while True:
		global steps
		steps=min(50,cur_step*2+2)
		density,stretch,bend = reset_sim(sim)
		st = time.time()
		loss = run_sim(steps, sim)
		en0 = time.time()
		# loss = get_loss(steps,sim)
		optimizer.zero_grad()
		print('step={}'.format(cur_step))
		print('forward time={}'.format(en0-st))
		print('loss={}'.format(loss.data))
		f.write('step {}: d={} loss={}\n'.format(cur_step, density.data*scaleden, loss.data))
		print('step {}: d={} loss={}\n'.format(cur_step, density.data*scaleden, loss.data))
		if loss < 1e-4:
			break
		#loss.backward()
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		#density.grad.data.clamp_(-1,1)
		#stretch.grad.data.clamp_(-1,1)
		#bend.grad.data.clamp_(-1e-3,1e-3)
		#print(density.grad)
		#print(stretch.grad)
		#print(bend.grad)
		optimizer.step(renew_loss)
		en1 = time.time()
		print('backward time={}'.format((en1-st)/steps))
		print(density*scaleden)
		print(stretch*scalestr)
		print(bend*scaleben)
		ld = (torch.tensor(matstd['density'],dtype=torch.float64)-density*scaleden).norm().data
		ls = (torch.tensor(matstd['stretching'],dtype=torch.float64)-stretch*scalestr).norm().data
		lb = (torch.tensor(matstd['bending'],dtype=torch.float64)-bend*scaleben).norm().data
		print('dis={} {} {}\n'.format(ld,ls,lb))
		matconfig['density'] = density.detach().numpy().tolist()
		matconfig['stretching'] = stretch.detach().numpy().tolist()
		matconfig['bending'] = bend.detach().numpy().tolist()
		save_config(matconfig, sys.argv[1]+'/mat.json')
		arcsim.delete_mesh(sim.cloths[0].mesh)
	get_dis(density, stretch, bend)

with open(sys.argv[1]+'/log.txt','w',buffering=1) as f:
	tot_step = 30
	sim=arcsim.get_sim()
	density,stretch,bend = reset_sim(sim)
	config['cloths'][0]['materials'][0]['reuse']=True
	save_config(config, sys.argv[1]+'/conf.json')
	lr = 0.1
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.LBFGS([density,stretch,bend],lr=lr,max_iter=20)
	#optimizer = torch.optim.SGD([{'params':density,'lr':lr}, {'params':stretch,'lr':lr}, {'params':bend,'lr': lr*100}],momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim)

print("done")

