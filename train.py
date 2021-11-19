import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from models import GDAN
from agent import run_agent

from mazeenv.mazeenv import MazeEnv
import datetime

import pdb
from setproctitle import setproctitle as ptitle
import time
import random
import os

from params import params

def run_sim(rank, shared_model, shared_optimizer, count, lock, goalStorage):
    # Set up logging
    if not os.path.exists('./'+params.weight_dir):
        os.mkdir('./'+params.weight_dir)
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # Change process title
    ptitle('Training {}'.format(rank))

    # Set GPU for current instance/process
    gpu_id = params.gpu_ids_train[rank % len(params.gpu_ids_train)]

    if shared_optimizer is None:
        print("\nshared_optimizer is None\n")

    # Set seed
    torch.manual_seed(params.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(params.seed + rank)

    maze_id = params.train_mazes[rank % len(params.train_mazes)]
    maze_path = params.mazes_path_root + str(maze_id)
    env = MazeEnv.load_vizdoom_env(maze_path, scaled_resolution=params.scaled_resolution,
                     living_reward=params.living_reward, target_reward=params.target_reward, non_target_penalty=params.non_target_penalty,
                     timeout_penalty=params.timeout_penalty, non_target_break=params.non_target_break)
    # Initialize model
    model = GDAN()
    with torch.cuda.device(gpu_id):
        model = model.cuda()

    # Initialize agent
    agent = run_agent(model, gpu_id, goalStorage)

    warmup = True
    i = 0
    while True:

        i += 1
        warmup = training(env, gpu_id, shared_model, agent, shared_optimizer, lock, count, warmup, rank, goalStorage)


def training(env, gpu_id, shared_model, agent, optimizer, lock, count, warmup, rank, goalStorage):

    next_obs = env.reset()
    obs = next_obs
    instruction_idx = env.get_target_idx()  # env.get_instruction()
    instruction = torch.from_numpy(np.array(instruction_idx)).view(1, -1)

    with torch.cuda.device(gpu_id):
        instruction = Variable(torch.LongTensor(instruction)).cuda()
    agent.synchronize(shared_model, instruction)
    agent.model.train()

    num_steps = 0

    done = False

    while not done:
        num_steps += 1
        before_obs = obs
        obs = next_obs

        if warmup:
            act = [random.randint(0, 2)]
        else:
            act, entropy, value, log_prob = agent.action_train(obs, instruction)

        next_obs, reward, done, info = env.step(act[0])

        if not warmup:
            agent.put_reward(reward, entropy, value, log_prob)

        if done:
            agent.done = True

            if reward >= 1.0 and num_steps > 1: # when before_obs is not the same with obs
                goalStorage.put(before_obs, instruction_idx)

            if warmup:
                if goalStorage.len() > params.minimum_warmup:
                    warmup = False
            else:
                if agent.get_reward_len() > 1:  # Avoid cases of success as soon as start
                    with lock:
                        count.value += 1
                    agent.training(next_obs, shared_model, optimizer, rank)
                else:
                    agent.clear_actions()
                break

    return warmup

