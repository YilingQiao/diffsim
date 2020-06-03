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
        with open('../conf/rigidcloth/manipulation/manipulation.json','r') as f:
            config = json.load(f)
    
        self.save_config(config, '../200120_manipulation/conf.json')

        spf        = config['frame_steps'] 
        self.steps = int(1*15*spf)

        self.observation_space     = np.zeros([13])
        self.action_space          = np.zeros([3])

        now = datetime.now()
        timestamp = datetime.timestamp(now)
        self.f = open('./log_ddpg_cloth%f.txt'%timestamp,'w',buffering=1)

        self.epoch = 0

    def reset(self):
        with torch.no_grad():
            arcsim.init_physics('../200120_manipulation/conf.json', '../200120_manipulation/out_ddpg',False)

            self.step_  = 0
            self.sim   = arcsim.get_sim()



            sigma = 0.4
            x = np.random.random()*sigma - 0.5*sigma + np.random.randint(2)*2-1
            self.goal = torch.tensor([0.0000, 0.0000, 0.0000,
            x, 
            0, 
            2+np.random.random()*sigma - 0.5*sigma],dtype=torch.float64)


            observation = []
            remain_time = torch.tensor([(self.steps)/50],dtype=torch.float64)
            observation = torch.cat([self.sim.obstacles[0].curr_state_mesh.dummy_node.x - self.goal,
                                        self.sim.obstacles[0].curr_state_mesh.dummy_node.v, 
                                        remain_time])
            return observation.numpy()

    def save_config(self, config, file):
        with open(file,'w') as f:
            json.dump(config, f)

    def step(self, action):
        with torch.no_grad():
            #sim.obstacles[0].curr_state_mesh.dummy_node.x = torch.tensor([0.0000, 0.0000, 0.0000,
            #np.random.random(), np.random.random(), -np.random.random()],dtype=torch.float64)
            action = torch.tensor(action, dtype=torch.float64)
            sim_input = torch.cat([torch.zeros([3], dtype=torch.float64), action])
            self.sim.obstacles[1].curr_state_mesh.dummy_node.v = sim_input 
            arcsim.sim_step()

            observation = []
            remain_time = torch.tensor([(self.steps - self.step_)/50],dtype=torch.float64)
            observation = torch.cat([self.sim.obstacles[0].curr_state_mesh.dummy_node.x - self.goal,
                                        self.sim.obstacles[0].curr_state_mesh.dummy_node.v, 
                                        remain_time])



            ans  = self.sim.obstacles[0].curr_state_mesh.dummy_node.x
            reward = self.get_loss(ans)

            self.step_ += 1
            
            done = False
            if self.step_ == self.steps:
                self.f.write('epoch {}: loss={}\n  ans = {}\n goal = {}\n'.format(self.epoch, (-reward).data,  ans.data, self.goal.data))
                done = True
                print("epoch %d-----------------------------  " % self.epoch)
                self.epoch += 1
                if self.epoch == 200:
                    quit()


            info = " "

            # print(observation.numpy())
            # print(reward.numpy())
            return observation.numpy(), reward.numpy(), done, info

    def get_loss(self, ans):
        #[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
        dif = ans - self.goal
        loss = torch.norm(dif.narrow(0, 3, 3), p=2)

        return -loss
