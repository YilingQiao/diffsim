import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os
import copy



with open('multibody.json','r') as f:
	config = json.load(f)

cube = config['obstacles'][0]

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

num_cube = 20

# for i in range(num_cube):
# 	new_cube = copy.deepcopy(cube)
# 	new_cube['transform']['translate'][1] -= 0.03 * (i+1)
# 	config['obstacles'].append(new_cube)
# 	print(config)
for i in range(num_cube):
	new_cube = copy.deepcopy(cube)
	new_cube['transform']['translate'][1] -= 0.03 * (i+1)
	config['obstacles'].append(new_cube)
	print(config)

save_config(config, "multibody_make.json".format(num_cube))


# save_config(matconfig, sys.argv[1]+'/mat.json')
# save_config(matconfig, sys.argv[1]+'/orimat.json')
# config['cloths'][0]['materials'][0]['data'] = sys.argv[1]+'/mat.json'
# config['end_time']=20
# save_config(config, sys.argv[1]+'/conf.json')


# torch.set_num_threads(8)
# spf = config['frame_steps']
# scalev=1

# def reset_sim(sim):
# 	arcsim.init_physics(sys.argv[1]+'/conf.json',sys.argv[1]+'/out',False)

# def get_loss(steps,sim):
# 	diffs = []
# 	for node0 in sim.cloths[0].mesh.nodes:
# 		diffs.append(node0.x)
# 	dis = torch.stack(diffs).mean(dim=0)
# 	dis[2] = dis[2] + 0.7
# 	print('dis=',dis)
# 	f.write('dis={}\n'.format(dis))
# 	return 0.5*torch.dot(dis,dis), dis

# def run_sim(steps,sim,vext):
# 	losses=[]
# 	for step in range(steps):
# 		sec = int(sim.frame/25)
# 		if sec < vext.shape[0]:
# 			for i in range(4):
# 				sim.cloths[0].mesh.nodes[i].v = sim.cloths[0].mesh.nodes[i].v + vext[sec,i]*scalev/spf
# 		arcsim.sim_step()
# 	return get_loss(steps,sim)

# def do_train(cur_step,optimizer,sim,vext):
# 	while True:
# 		steps=4*25*spf
# 		reset_sim(sim)
# 		st = time.time()
# 		loss, dis = run_sim(steps, sim, vext)
# 		en0 = time.time()
# 		# loss = get_loss(steps,sim)
# 		optimizer.zero_grad()
# 		print('step={}'.format(cur_step))
# 		print('forward time={}'.format(en0-st))
# 		print('loss={}'.format(loss.data))
# 		f.write('step {}: loss={}\n'.format(cur_step, loss.data))
# 		print('step {}: loss={}\n'.format(cur_step, loss.data))
# 		loss.backward()
# 		vext.grad.data.clamp_(-10,10)
# 		if dis[:2].norm() < 0.5 and dis[2]>0 and dis[2]<0.5:#loss<1e-3:#
# 			break
# 		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
# 		en1 = time.time()
# 		print('backward time={}'.format(en1-en0))
# 		print(vext.grad)
# 		optimizer.step()
# 		print(vext*scalev)
# 		#f.write('{}\n'.format(vext.grad));
# 		f.write('{}\n'.format(vext*scalev));
# 		arcsim.delete_mesh(sim.cloths[0].mesh)
# 		# break

# with open(sys.argv[1]+'/log.txt','w',buffering=1) as f:
# 	tot_step = 1
# 	sim=arcsim.get_sim()
# 	reset_sim(sim)
# 	vext = torch.zeros([3,4,3],dtype=torch.float64,requires_grad=True)
# 	vext = torch.tensor(
# 		[[[ 0.9664, -0.0545, 19.5551],
#          [ 1.2179, -0.3708, 19.5542],
#          [ 1.1454, -0.0323, 19.5542],
#          [ 1.1271,  0.0557, 19.5552]],

#         [[ 0.6964, -0.0535, 19.9428],
#          [ 1.0035, -0.0466, 19.9417],
#          [ 0.9208, -0.1224, 19.9398],
#          [ 0.6681, -0.1472, 19.9410]],

#         [[ 0.4484, -0.0576, 18.6777],
#          [ 0.5289, -0.0813, 18.6886],
#          [ 0.4961, -0.0276, 18.6901],
#          [ 0.4655, -0.0455, 18.6876]]],
# dtype=torch.float64,requires_grad=True)
# 	config['cloths'][0]['materials'][0]['reuse']=True
# 	save_config(config, sys.argv[1]+'/conf.json')
# 	lr = 0.01
# 	momentum = 0.7
# 	f.write('lr={} momentum={}\n'.format(lr,momentum))
# 	optimizer = torch.optim.SGD([{'params':vext,'lr':lr}],momentum=momentum)
# 	# optimizer = torch.optim.Adadelta([density, stretch, bend])
# 	for cur_step in range(tot_step):
# 		do_train(cur_step,optimizer,sim,vext)

# print("done")

