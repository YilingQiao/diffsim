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
        with open('../conf/rigidcloth/drag/drag.json','r') as f:
            config = json.load(f)
    
        if not os.path.exists('../200121_drag/'):
            os.mkdir('../200121_drag/')

        self.save_config(config, '../200121_drag/conf.json')

        spf        = config['frame_steps'] 
        self.steps = 15#int(1*25*spf)

        self.observation_space     = np.zeros([28])
        self.action_space          = np.zeros([3])


        now = datetime.now()
        timestamp = datetime.timestamp(now)
        self.f = open('./log_ddpg_cloth%f.txt'%timestamp,'w',buffering=1)
        self.epoch = 0

    def reset(self):
        with torch.no_grad():

            self.step_  = 0


            sigma = 0.1
            z = np.random.random()*sigma + 0.4

            y = np.random.random()*sigma - sigma/2
            x = np.random.random()*sigma - sigma/2


            ini_co = torch.tensor([0.0000, 0.0000, 0.0000, 0.4744, 0.4751, 0.0064], dtype=torch.float64)
            goal = torch.tensor([0.0000, 0.0000, 0.0000,
             0, 0, z],dtype=torch.float64)
            goal = goal + ini_co

            self.goal = goal

            if self.epoch % 4==0 and self.epoch <= 60:
                arcsim.init_physics('../200121_drag/conf.json','../200121_drag/out_ddpg%d'%self.epoch,False)
                text_name = '../200121_drag/out_ddpg%d'%self.epoch+ "/goal.txt"
            
                np.savetxt(text_name, goal[3:6], delimiter=',')
            else:
                arcsim.init_physics('../200121_drag/conf.json', '../200121_drag/out_ddpg',False)

            self.sim   = arcsim.get_sim()

            handles = [0,1,2,3]
            observation = []
            remain_time = torch.tensor([(self.steps)/50],dtype=torch.float64)
            for i in range(len(handles)):
                observation.append(self.sim.cloths[0].mesh.nodes[handles[i]].x)
                observation.append(self.sim.cloths[0].mesh.nodes[handles[i]].v)



            dis = self.sim.obstacles[0].curr_state_mesh.dummy_node.x - self.goal
            observation.append(dis.narrow(0, 3, 3))
            observation.append(remain_time)
            observation = torch.cat(observation)

            # observation = []
            # remain_time = torch.tensor([(self.steps)/50],dtype=torch.float64)
            # observation = torch.cat([self.sim.obstacles[0].curr_state_mesh.dummy_node.x - self.goal,
            #                             self.sim.obstacles[0].curr_state_mesh.dummy_node.v, 
            #                             remain_time])
            return observation.detach().numpy()

    def save_config(self, config, file):
        with open(file,'w') as f:
            json.dump(config, f)

    def step(self, action):
        with torch.no_grad():
            #sim.obstacles[0].curr_state_mesh.dummy_node.x = torch.tensor([0.0000, 0.0000, 0.0000,
            #np.random.random(), np.random.random(), -np.random.random()],dtype=torch.float64)
            action = torch.tensor(action, dtype=torch.float64)

            
            handles = [0,1,2,3]
            for i in range(len(handles)):
                self.sim.cloths[0].mesh.nodes[handles[i]].v = action


            arcsim.sim_step()

            observation = []
            remain_time = torch.tensor([(self.steps - self.step_)/50],dtype=torch.float64)
            for i in range(len(handles)):
                observation.append(self.sim.cloths[0].mesh.nodes[handles[i]].x)
                observation.append(self.sim.cloths[0].mesh.nodes[handles[i]].v)



            dis = self.sim.obstacles[0].curr_state_mesh.dummy_node.x - self.goal
            observation.append(dis.narrow(0, 3, 3))
            observation.append(remain_time)
            observation = torch.cat(observation)


            ans  = self.sim.obstacles[0].curr_state_mesh.dummy_node.x
            reward = self.get_loss(ans)

            self.step_ += 1
            
            done = False
            if self.step_ == self.steps:
                self.f.write('epoch {}: loss={}\n  ans = {}\n goal = {}\n'.format(self.epoch, (-reward).data,  ans.data, self.goal.data))
                done = True
                print("epoch %d-----------------------------  " % self.epoch)
                self.epoch += 1
                if self.epoch > 200:
                    quit()


            info = " "

            # print(observation.numpy())
            # print(reward.numpy())
            return observation.detach().numpy(), reward.detach().numpy(), done, info

    def get_loss(self, ans):
        #[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
        dif = ans - self.goal
        loss = torch.norm(dif.narrow(0, 3, 3), p=2)

        return -loss
