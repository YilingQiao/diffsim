import torch
import arcsim
import json
import sys
import gc
import os
import copy
import numpy as np
import time

def reset_sim():
	arcsim.init_physics('conf/rigidcloth/scale/scale_make.json','',False)


def run_sim():
	for step in range(20):
		arcsim.sim_step()
		print(step)

perline = 50

def make_json (scale = 1):
	with open('conf/rigidcloth/scale/scale.json','r') as f:
		config = json.load(f)

	config['cloths'][0]["transform"]["scale"] = scale

	def save_config(config, file):
		with open(file,'w') as f:
			json.dump(config, f)

	save_config(config, "conf/rigidcloth/scale/scale_make.json")

print(sys.argv)

index = int(sys.argv[1])
scale = index
print("scale")
print(scale)
make_json(scale)


#arcsim.msim(4,['arcsim','simulateoffline','conf/rigidcloth/multibody/multibody_make.json','out'])
# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/scale/scale_make.json','out'])
reset_sim()
t = time.time()
run_sim()

time_record = time.time() - t



print(time_record)

f = open("scale.txt", "a")
f.write("%d %f\n"%(scale, time_record))