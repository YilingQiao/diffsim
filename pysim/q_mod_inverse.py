import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os
import numpy as np
from datetime import datetime

class EnvMan(object):
    """docstring for ClassName"""
    def __init__(self):

        with open('../conf/rigidcloth/bounce/bounce.json','r') as f:
            config = json.load(f)
    
        if not os.path.exists('../200121_bounce/'):
            os.mkdir('../200121_bounce/')
        self.save_config(config, '../200121_bounce/conf.json')

        spf        = config['frame_steps'] 
        self.steps = 20

        self.observation_space     = np.zeros([13])
        self.action_space          = np.zeros([100])

        now = datetime.now()
        timestamp = datetime.timestamp(now)
        self.f = open('./log_cmaes%f.txt'%timestamp,'w',buffering=1)
        self.epoch = 0

    def reset(self):
        with torch.no_grad():
            arcsim.init_physics('../200121_bounce/conf.json', '../200121_bounce/out_cmaes',False)
            self.step_  = 0
            self.sim   = arcsim.get_sim()

    def save_config(self, config, file):
        with open(file,'w') as f:
            json.dump(config, f)

    def step(self, param_g):

        with torch.no_grad():
            param_g = torch.tensor(param_g, dtype=torch.float64)
            param_g = param_g.view(20, 5)
            # dx = []
            # dx.append(torch.zeros([3],dtype=torch.float64))
            # dx.append(param_g)
            # dx.append(torch.zeros([1],dtype=torch.float64))
            # dx = torch.cat(dx)
            # self.sim.obstacles[0].curr_state_mesh.dummy_node.x += dx
            # sim.obstacles[2].curr_state_mesh.dummy_node.x = param_g[1]
            for step in range(20):

                dx = []
                # dx.append(torch.zeros([3],dtype=torch.float64))
                dx.append(param_g[step])
                dx.append(torch.zeros([1],dtype=torch.float64))
                dx = torch.cat(dx)
                self.sim.obstacles[0].curr_state_mesh.dummy_node.v += dx

                arcsim.sim_step()

            cnt = 0

            ans = self.sim.obstacles[0].curr_state_mesh.dummy_node.x
            ans = ans

            # ans = torch.tensor([0, 0, 0],dtype=torch.float64)
            # for node in sim.cloths[0].mesh.nodes:
            #   cnt += 1
            #   ans = ans + node.x
            # ans /= cnt
            loss = self.get_loss(ans, param_g)

            self.f.write('epoch {}: loss={}\n \n '.format(self.epoch, loss.data))

            print("epoch %d-----------------------------  " % self.epoch)

            self.epoch += 1

            return loss.detach().numpy()

    def get_loss(self, ans, param_g):
        #[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
        vec = torch.tensor([0, 0],dtype=torch.float64)
        loss = torch.norm(ans.narrow(0, 3, 2) - vec, p=2)
        reg  = torch.norm(param_g, p=2)*0.05

        return loss + reg
