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
matfile = './'+config['cloths'][0]['materials'][0]['data']
matfile = './wangout/'+sys.argv[4]+'/mat.json'
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

scaleden = 1
scalestr = 1e3
scaleben = 1e-4

def reset_sim(sim):
	arcsim.init_physics(sys.argv[1]+'/conf.json','',False)
	mat = sim.cloths[0].materials[0]
	density = mat.densityori
	stretch = mat.stretchingori
	bend = mat.bendingori
	return density, stretch, bend

def get_loss_eval(steps,sim):
	fstd = sys.argv[2]+'/%04d_00.obj'%(steps/2)
	mesh = arcsim.Mesh()
	arcsim.load_obj(mesh, fstd)
	diffs = []
	for node0,node1 in zip(mesh.nodes, sim.cloths[0].mesh.nodes):
		diffs.append((node0.x-node1.x).norm())
	arcsim.delete_mesh(mesh)
	return torch.stack(diffs).mean()

def run_sim(steps,sim):
	losses=[]
	for step in range(steps):
		arcsim.sim_step()
		if step % 2 == 1:
			losses.append(get_loss_eval(step+1,sim))
	return torch.stack(losses).mean()

def get_dis(density, stretch, bend):
	with torch.no_grad():
		f.write('grads:\n')
		f.write('{}\n'.format(density*scaleden))
		f.write('{}\n'.format(stretch*scalestr))
		f.write('{}\n'.format(bend*scaleben))
		stdd = torch.tensor(matstd['density'],dtype=torch.float64)
		stds = torch.tensor(matstd['stretching'],dtype=torch.float64)
		stdb = torch.tensor(matstd['bending'],dtype=torch.float64)
		# print(stretch[0]*scalestr-stds[0],stds[0])
		ld = (density*scaleden-stdd).norm()/stdd.norm()
		ls = (stretch*scalestr-stds).norm()/stds.norm()
		ls0 = (stretch[0]*scalestr-stds[0]).norm()/stds[0].norm()
		lb = (bend*scaleben-stdb).norm()/stdb.norm()
		print('dis={} {} {} {}\n'.format(ld,ls,lb,ls0))
		f.write('dis={} {} {} {}\n'.format(ld,ls,lb,ls0))

def naive_guess(density,stretchori,bend):
	return density,stretchori,bend
	M=[]
	for i in range(5,25):
		fstd = sys.argv[2]+'/%04d_00.obj'%(i)
		mesh=arcsim.Mesh()
		arcsim.load_obj(mesh, fstd)
		fstd = sys.argv[2]+'/%04d_00.obj'%(i+1)
		arcsim.load_obj(sim.cloths[0].mesh, fstd)
		arcsim.compute_ms_data(sim.cloths[0].mesh)
		for n0 in sim.cloths[0].mesh.nodes:
			n0.m = torch.zeros([],dtype=torch.float64)

		n = len(sim.cloths[0].mesh.nodes)
		fext = torch.zeros([n,3],dtype=torch.float64)
		Jext = torch.zeros([n,3],dtype=torch.float64)
		arcsim.add_external_forces(sim.cloths[0], sim.gravity, sim.wind, fext, Jext);
		m=[]
		i=0
		for n0,n1 in zip(mesh.nodes,sim.cloths[0].mesh.nodes):
			a=(n1.v-n0.v)/config['frame_time']*config['frame_steps']
			a[2] += 9.8
			if i!=2 or i!=3:
				m.append((fext[i].sum()/a.sum()))
			i+=1
		arcsim.delete_mesh(mesh)
		M.append(torch.relu(torch.stack(m).sum()))
	print('density={}'.format(torch.stack(M).mean()))
	density = torch.stack(M).mean()/scaleden

	mesh=arcsim.Mesh()
	fstd = sys.argv[2]+'/%04d_00.obj'%(4)
	arcsim.load_obj(mesh,fstd)
	mini = 1e10
	maxi = -1e10
	for n0 in mesh.nodes:
		mini = min(mini, n0.x[2])
		maxi = max(maxi, n0.x[2])
	defo = maxi-mini-1
	stre = density*9.8/defo*2
	arcsim.delete_mesh(mesh)
	print(stre)
	stretch = torch.tensor([[1,0,1,0]],dtype=torch.float64).repeat([6,1])*stre/scalestr
	# stretch[:,0] = stretchori[:,0]
	stretch[:,1] = stretchori[:,1]
	# stretch[:,2] = stretchori[:,2]
	stretch[:,3] = stretchori[:,3]
	print(stretch)

	bend = bend

	matconfig['density'] = density.detach().numpy().tolist()
	matconfig['stretching'] = stretch.detach().numpy().tolist()
	matconfig['bending'] = bend.detach().numpy().tolist()
	save_config(matconfig, sys.argv[1]+'/mat.json')
	return density,stretch,bend

with open(sys.argv[1]+'/log_prior.txt','w',buffering=1) as f:
	sim=arcsim.get_sim()
	density,stretchori,bend = reset_sim(sim)
	save_config(config, sys.argv[1]+'/conf.json')

	density,stretch,bend=naive_guess(density,stretchori,bend)

	get_dis(density, stretch, bend)
	density,stretchori,bend = reset_sim(sim)
	loss = run_sim(100, sim)
	print('loss=',loss)
	f.write('loss={}\n'.format(loss))


print("done")

