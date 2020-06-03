#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import pickle

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))

import numpy as np
import gym
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box
import torch
import arcsim
import os
import json
import time

argv1="./eval_ppo"
if not os.path.exists(argv1):
    os.mkdir(argv1)

with open('conf/demo_throw_test.json','r') as f:
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


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=3, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    return parser


def run(args, parser):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    rollout(agent, args.env, num_steps, args.out, args.no_render)


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def rollout(agent, env_name, num_steps, out=None, no_render=True):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "local_evaluator"):
        # env = agent.local_evaluator.env
        env = SimpleCorridor({})
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.local_evaluator.multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.local_evaluator.policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        # env = gym.make(env_name)
        env = SimpleCorridor({})
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if out is not None:
        rollouts = []
    steps = 0
    while steps < (num_steps or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        if out is not None:
            rollout = []
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if out is not None:
                rollout.append([obs, action, next_obs, reward, done])
            steps += 1
            obs = next_obs
        if out is not None:
            rollouts.append(rollout)
        print("Episode reward", reward_total)

    if out is not None:
        pickle.dump(rollouts, open(out, "wb"))


if __name__ == "__main__":
    ModelCatalog.register_custom_model("my_model", CustomModel)
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
