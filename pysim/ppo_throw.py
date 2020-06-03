"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.tune import grid_search

import torch
import arcsim
import os
import json
import time

argv1="./control_ppo"
if not os.path.exists(argv1):
    os.mkdir(argv1)

with open('conf/demo_throw.json','r') as f:
    config = json.load(f)
matfile = config['cloths'][0]['materials'][0]['data']
with open(matfile,'r') as f:
    matconfig = json.load(f)

def save_config(config, file):
    with open(file,'w') as f:
        json.dump(config, f)

save_config(matconfig, argv1+'/mat.json')
save_config(matconfig, argv1+'/orimat.json')
config['cloths'][0]['materials'][0]['data'] = argv1+'/mat.json'
config['end_time']=20
save_config(config, argv1+'/conf.json')


torch.set_num_threads(8)
spf = config['frame_steps']
scalev=1

def reset_sim():
    arcsim.init_physics(argv1+'/conf.json',argv1+'/out',False)

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        sim = arcsim.get_sim()
        self.action_space = Box(low=-30.,high=30.,shape=(4*3,),dtype=np.float64)
        self.observation_space = Box(
            low=-100.0, high=100.0, shape=(81*2*3,), dtype=np.float64)

    def get_obs(self):
        xs=[]
        sim = arcsim.get_sim()
        for node in sim.cloths[0].mesh.nodes:
            xs.append(torch.stack([torch.tensor(arcsim.tovec(node.x),dtype=torch.float64),torch.tensor(arcsim.tovec(node.v),dtype=torch.float64)]))
        ans = torch.stack(xs).flatten().squeeze()
        return ans.detach().numpy()

    def get_loss(self):
        diffs = []
        sim = arcsim.get_sim()
        for node0 in sim.cloths[0].mesh.nodes:
            diffs.append(torch.tensor(arcsim.tovec(node0.x),dtype=torch.float64))
        dis = torch.stack(diffs).mean(dim=0)
        dis[2] = dis[2] + 0.7
        return dis.norm().detach().item()

    def reset(self):
        reset_sim()
        return self.get_obs()

    def step(self, action):
        sim = arcsim.get_sim()
        print("step!",sim.step)
        print(action.reshape([4,3]))
        for _ in range(spf*25):
            sec = int(sim.frame/25)
            if sec < 3:
                for i in range(4):
                    for k in range(3):
                        sim.cloths[0].mesh.nodes[i].v[k] = sim.cloths[0].mesh.nodes[i].v[k] + action[i*3+k]*scalev/spf
            arcsim.sim_step()
        sec = int(sim.frame/25)
        if sec == 3:
            for _ in range(spf*25):
                arcsim.sim_step()
        done = sim.step >= 4*25*spf
        return self.get_obs(), -self.get_loss() if done else 0, done, {}


class CustomModel(Model):
    """Example of a custom model.
    This model just delegates to the built-in fcnet.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        self.obs_in = input_dict["obs"]
        self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
                                           self.action_space, num_outputs,
                                           options)
        return self.fcnet.outputs, self.fcnet.last_layer


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    object_store_memory = int(0.6 * ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9)
    ray.init(
                include_webui=False,
                ignore_reinit_error=True,
                # plasma_directory="/tmp",
                object_store_memory=object_store_memory,
            )
    ModelCatalog.register_custom_model("my_model", CustomModel)
    sim=arcsim.get_sim()
    reset_sim()
    save_config(config, argv1+'/conf.json')
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 40000,
        },
        config={
            "env": SimpleCorridor,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "lr": grid_search([1e-3]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
            },
            "sample_batch_size":1,
            "train_batch_size":20,
            "sgd_minibatch_size":20,
        },
        local_dir=argv1,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        resume="prompt",
    )

