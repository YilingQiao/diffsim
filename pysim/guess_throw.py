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

with open('conf/demo_throw_test.json','r') as f:
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
spf = config['frame_steps']
scalev=1

def reset_sim(sim):
	arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out',False)

def get_loss(steps,sim):
	diffs = []
	for node0 in sim.cloths[0].mesh.nodes:
		diffs.append(node0.x)
	dis = torch.stack(diffs).mean(dim=0)
	dis[2] = dis[2] + 0.7
	print('dis=',dis)
	f.write('dis={}\n'.format(dis))
	return 0.5*torch.dot(dis,dis), dis

def get_xv(sim):
	xs = []
	vs = []
	for node0 in sim.cloths[0].mesh.nodes:
		xs.append(node0.x)
		vs.append(node0.v)
	x = torch.stack(xs).mean(dim=0)
	v = torch.stack(vs).mean(dim=0)
	return x, v

def run_sim(steps,sim):
	losses=[]
	vext = torch.zeros([4,3],dtype=torch.float64)
	for step in range(steps):
		sec = int(sim.frame/25)
		if sec < 3:
			if sim.frame % 25 == 0 and step % spf == 0:
				x,v = get_xv(sim)
				target = torch.tensor([0,0,-0.7],dtype=torch.float64)
				target -= x + v*(4-sec)
				target[2] += 0.5*9.8*(4-sec)*(4-sec)
				target /= 4-sec-0.5
				vext = target
				vext = vext.unsqueeze(0).repeat([4,1])
				print("vext={}\n".format(vext))
				f.write("vext={}\n".format(vext))
			for i in range(4):
				sim.cloths[0].mesh.nodes[i].v = sim.cloths[0].mesh.nodes[i].v + vext[i]*scalev/spf
		arcsim.sim_step()
	return get_loss(steps,sim)

def do_train(cur_step,sim):
	while True:
		steps=4*25*spf
		reset_sim(sim)
		loss, dis = run_sim(steps, sim)
		# loss = get_loss(steps,sim)
		print('step={}'.format(cur_step))
		print('loss={}'.format(loss.data))
		f.write('step {}: loss={}\n'.format(cur_step, loss.data))
		print('step {}: loss={}\n'.format(cur_step, loss.data))
		#f.write('{}\n'.format(vext.grad));
		arcsim.delete_mesh(sim.cloths[0].mesh)
		break

with open(sys.argv[1]+'/log.txt','w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	reset_sim(sim)
	config['cloths'][0]['materials'][0]['reuse']=True
	save_config(config, sys.argv[1]+'/conf.json')
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,sim)

print("done")

