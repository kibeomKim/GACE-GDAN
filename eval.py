import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs --bind_all

from models import GDAN
from agent import run_agent
from params import params
import datetime

from setproctitle import setproctitle as ptitle
import time
import numpy as np
import os

import gym
import multitarget_visnav


def test(rank, shared_model, shared_optimizer, count, lock):
    # Set up logging
    if not os.path.exists('./'+params.weight_dir):
        os.mkdir('./'+params.weight_dir)
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # Change process title
    ptitle('Testing {}'.format(rank))

    # Set GPU for current instance/process
    gpu_id = params.gpu_ids_test[rank % len(params.gpu_ids_test)]

    if shared_optimizer is None:
        print("\nshared_optimizer is None\n")

    # Set seed
    torch.manual_seed(params.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(params.seed + rank)

    # Load Vizdoom environment
    maze_id = params.eval_mazes[rank % len(params.train_mazes)]
    env = gym.make(maze_id, scaled_resolution=params.scaled_resolution,
                   living_reward=params.living_reward,
                   goal_reward=params.goal_reward,
                   non_goal_penalty=params.non_goal_penalty,
                   timeout_penalty=params.timeout_penalty,
                   non_goal_break=params.non_goal_break)

    # Initialize model
    model = GDAN()
    with torch.cuda.device(gpu_id):
        model = model.cuda()

    # Initialize agent
    agent = run_agent(model, gpu_id)
    now = datetime.datetime.now()
    nowDate = now.strftime('%Y-%m-%d-%H:%M:%S')
    writer = SummaryWriter('runs/GDAN_{}_{}'.format(params.map, maze_id, nowDate))

    best_rate = 0.0
    save_model_index = 0

    while True:
        with lock:
            n_update = count.value
        with torch.cuda.device(gpu_id):
            agent.model.load_state_dict(shared_model.state_dict())

        start_time = time.time()
        best_rate, save_model_index = testing(rank, env, maze_id, gpu_id, agent, n_update, best_rate,
                                              save_model_index, start_time, writer, shared_optimizer)


def testing(rank, env, maze_id, gpu_id, agent, n_update, best_rate, save_model_index, start_time,
            writer, optimizer):
    evals = []
    agent.model.eval()

    for i in range(params.n_eval):
        next_obs = env.reset()
        instruction = env.get_goal_idx()

        instruction = torch.from_numpy(np.array(instruction)).view(1, -1)

        with torch.cuda.device(gpu_id):
            instruction = Variable(torch.LongTensor(instruction)).cuda()
            agent.cx = Variable(torch.zeros(1, 256).cuda())
            agent.hx = Variable(torch.zeros(1, 256).cuda())
        agent.eps_len = 0

        step, total_rew, good = 0, 0, 0
        done = False

        while not done:
            obs = next_obs
            act = agent.action_test(obs, instruction)

            next_obs, rew, done, info = env.step(act[0])
            total_rew += rew

            if rew >= 1.0:  # success
                good = 1

            step += 1

            if done:
                break

        evals.append((step, total_rew, good))

    if len(evals) > 0:
        success = [e for e in evals if e[2] > 0]
        success_rate = (len(success) / len(evals)) * 100

        if success_rate >= best_rate:
            best_rate = success_rate
            print("***test rank {}, save {}".format(rank, str(n_update)))
            with torch.cuda.device(gpu_id):
                save_dict = {
                    'n_update': n_update,
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'success_rate': success_rate,
                }
                save_path = params.weight_dir + 'model' + str(n_update) + '.ckpt'
                torch.save(save_dict, save_path)
            save_model_index += 1

        avg_reward = sum([e[1] for e in evals]) / len(evals)
        avg_length = sum([e[0] for e in evals]) / len(evals)
        msg = ' '.join([
            "++++++++++ Task Stats +++++++++++\n",
            "Time {}\n".format(time.strftime("%dd %Hh %Mm %Ss", time.gmtime(time.time() - start_time))),
            "Episode Played: {:d}\n".format(len(evals)),
            "N_Update = {:d}\n".format(n_update),
            "Maze id: {}\n".format(maze_id),
            "Avg Reward = {:5.3f}\n".format(avg_reward),
            "Avg Length = {:.3f}\n".format(avg_length),
            "Best rate {:3.2f}, Success rate {:3.2f}%".format(best_rate, success_rate)
        ])
        writer.add_scalar('successRate/maze {}'.format(maze_id), success_rate / 100., n_update)
        writer.add_scalar('avgReward/maze {}'.format(maze_id), avg_reward, n_update)
        writer.add_scalar('avgLength/maze {}'.format(maze_id), avg_length, n_update)
        # print(msg)

    return best_rate, save_model_index
