import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(81*2*3,200)
        self.fc2 = nn.Linear(200, 4*3)

    def forward(self, x):
        x = x.view(1, 81*2*3)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

print(sys.argv)#prefix
if not os.path.exists(sys.argv[1]):
	os.mkdir(sys.argv[1])

with open('conf/demo_throw.json','r') as f:
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

def run_sim(steps,sim,model):
	losses=[]
	vext = []
	for step in range(steps):
		sec = int(sim.frame/25)
		if sec < 3:
			if sim.frame % 25 == 0 and step % spf == 0:
				inp = []
				for n in sim.cloths[0].mesh.nodes:
					inp += [n.x, n.v]
				v = model(torch.stack(inp)/100).reshape([4,3])*30
				vext.append(v)
				print(v)
			for i in range(4):
				sim.cloths[0].mesh.nodes[i].v = sim.cloths[0].mesh.nodes[i].v + v[i]*scalev/spf
		arcsim.sim_step()
	a,b = get_loss(steps,sim)
	return a,b,torch.stack(vext)

def do_train(cur_step,optimizer,sim,model):
	model.train()
	while True:
		steps=4*25*spf
		reset_sim(sim)
		st = time.time()
		loss, dis, vext = run_sim(steps, sim, model)
		en0 = time.time()
		# loss = get_loss(steps,sim)
		optimizer.zero_grad()
		print('step={}'.format(cur_step))
		print('forward time={}'.format(en0-st))
		print('loss={}'.format(loss.data))
		f.write('step {}: loss={}\n'.format(cur_step, loss.data))
		print('step {}: loss={}\n'.format(cur_step, loss.data))
		loss.backward()
		nn.utils.clip_grad_value_(model.parameters(), 10)
		if dis[:2].norm() < 0.5 and dis[2]>0 and dis[2]<0.5:#dis[2]<0.5:
			break
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		en1 = time.time()
		print('backward time={}'.format(en1-en0))
		optimizer.step()
		#f.write('{}\n'.format(vext.grad));
		f.write('{}\n'.format(vext*scalev));
		arcsim.delete_mesh(sim.cloths[0].mesh)
		# break
	torch.save(model.state_dict(),sys.argv[1]+"/3.pt")

with open(sys.argv[1]+'/log.txt','w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	reset_sim(sim)
	model = Net().double();
	model.load_state_dict(torch.load(sys.argv[1]+"/2.pt"))
	config['cloths'][0]['materials'][0]['reuse']=True
	save_config(config, sys.argv[1]+'/conf.json')
	lr = 0.0001
	momentum = 0.5
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,model)

print("done")

