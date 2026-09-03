# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.


import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import gymnasium as gym
import numpy as np
from utilis.config import ARGConfig
from utilis.default_config import default_config
from utilis.logger import Logger
from model.algo import flowAC
from utilis.Replaybuffer import ReplayMemory
import datetime
import itertools
from time import time
from envs.dm_control import make_env
# from humanoid_bench.env import ROBOTS, TASKS

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True

def evaluation(agent, env, total_numsteps, logger):
    avg_reward = 0.
    avg_success = 0.
    for _  in range(config.eval_times):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
        if 'solved' in info.keys():
            avg_success += float(info['solved'])
        elif 'success' in info.keys():
            avg_success += float(info['success'])
    avg_reward /= config.eval_times
    avg_success /= config.eval_times

    logger.log({
        "step": total_numsteps,
        "reward": avg_reward,
        "seed": config.seed,
    }, "eval")

    return avg_reward

def train_loop(config):
    # for humanoid_bench
    # env = gym.make(config.task)
    # for dm control
    env = make_env(config)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    env.action_space.seed(config.seed)
    np.random.seed(config.seed)
    # set seed
    # Agent
    agent = flowAC(env.observation_space.shape[0], env.action_space, config)

    result_path = './results/{}/{}/{}/{}_{}_{}'.format(config.task, config.algo, config.exp_name,
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                      config.policy, config.seed)

    checkpoint_path = result_path + '/' + 'checkpoint'
    # training logs
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open(os.path.join(result_path, "config.log"), 'w') as f:
        f.write(str(config))

    logger = Logger(result_path, save_csv={"train": False, "eval": True})

    # memory
    memory = ReplayMemory(config.replay_size, config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_reward = -1e6
    start_time = time()
    ep_idx = 0

    if config.eval is True:
        avg_reward = evaluation(agent, env, total_numsteps, logger)
        if avg_reward >= best_reward and config.save is True:
            best_reward = avg_reward
            agent.save_checkpoint(checkpoint_path, 'best')

    for _ in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset()
        episode_success = 0.0

        while not done:
            if config.start_steps > total_numsteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)  # Sample action from policy

            if config.start_steps <= total_numsteps:
                # Number of updates per step in environment
                for i in range(config.updates_per_step):
                    # Update parameters of all the networks
                    agent.update_parameters(memory, config.batch_size, updates)
                    updates += 1

            next_state, reward, done, truncated, info = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            state = next_state

            # test agent
            if total_numsteps % config.eval_numsteps == 0 and config.eval is True:
                avg_reward = evaluation(agent, env, total_numsteps, logger)
                if avg_reward >= best_reward and config.save is True:
                    best_reward = avg_reward
                    agent.save_checkpoint(checkpoint_path, 'best')

        if total_numsteps > config.num_steps:
            break

        ep_idx += 1
        if 'success' in info:
            episode_success = float(info['success'])
        elif 'solved' in info:
            episode_success = float(info['solved'])

        logger.log({
            "step": total_numsteps,
            "episode": ep_idx,
            "reward": episode_reward,
            "seed": config.seed,
            "total_time": time() - start_time,
        }, "train")

    env.close()
    logger.finish()



if __name__ == "__main__":
    config = default_config

    arg = ARGConfig()
    arg.add_arg("task", config.task, "Environment name (domain-task)")
    arg.add_arg("device", config.device, "Computing device")
    arg.add_arg("algo", config.algo, "choose algo")
    arg.add_arg("exp_name", config.exp_name, "Experiment name")
    arg.add_arg("seed", config.seed, "experiment seed")
    arg.add_arg("epsilon", config.epsilon, "random noise for exploration")
    arg.add_arg("lamda", config.lamda, "lagrange_multiplier")
    arg.add_arg("action_repeat", config.action_repeat, "Action repeat for DMControl")
    arg.add_arg("num_steps", config.num_steps, "Total environment steps")
    arg.add_arg("eval_numsteps", config.eval_numsteps, "Evaluation interval in environment steps")
    arg.parser()

    config.update(arg)
    print(f">>>> Training {config.algo} on {config.task} environment, on {config.device}")
    train_loop(config)
