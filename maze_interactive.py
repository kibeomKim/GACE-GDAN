from mazeenv.mazeenv import MazeEnv
import numpy as np
from itertools import count
import cv2
import os


output_folder = './interactive_img_dbg'

params = {
    'maze_path': './maps/V1/0',
    'living_reward': -0.0025,
    'target_reward': 10.0,
    'non_target_penalty': 1.0,
    'non_target_break': True,
    'timeout_penalty': 0.1
}

targets = ['Card', 'Armor', 'Skull', 'Bonus']

if __name__ == '__main__':

    env = MazeEnv.load_vizdoom_env(mazes_path=params['maze_path'],
        scaled_resolution=(240, 320), living_reward=params['living_reward'],
        target_reward=params['target_reward'],
        non_target_penalty=params['non_target_penalty'],
        timeout_penalty=params['timeout_penalty'],
        non_target_break=params['non_target_break'])

    os.makedirs(output_folder, exist_ok=True)

    obs = env.reset()
    target = targets[env.get_target_idx()]
    print('start, target {}'.format(target))
    total_reward = 0.0
    convert_img = True
    save_img_idx = 0
    for t in count():
        if convert_img:
            obs = obs[-1,:-1]
            obs = np.transpose(obs, (1, 2, 0))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
            cv2.imshow('window', obs)

        convert_img = True
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('1'):
            obs, reward, dones, info = env.step(0)
        elif key == ord('2'):
            obs, reward, dones, info = env.step(1)
        elif key == ord('3'):
            obs, reward, dones, info = env.step(2)
        elif key == ord('r'):
            obs = env.reset()
            target = targets[env.get_target_idx()]
            print('manual reset, target {}'.format(target))
            continue
        elif key == ord('s'):
            path = '{}/{}.png'.format(output_folder, save_img_idx)
            cv2.imwrite(path, obs)
            print('image saved to path {}'.format(path))
            save_img_idx += 1
            convert_img = False
            continue
        else:
            convert_img = False
            continue
        total_reward += reward
        print('total reward: {}, dones: {}, info: {}'.format(total_reward, dones, info))
        if dones:
            obs = env.reset()
            target = targets[env.get_target_idx()]
            print('new episode, target {}'.format(target))

    env.close()
