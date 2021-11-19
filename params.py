from collections import OrderedDict
import time
import os
import random
import datetime

class Params:
    def __init__(self):
        self.map = 'V1'  # or 'V2' 'V3' 'V4'

        # Configs for running trains/test
        self.num_train_processes = 20
        self.train_mazes = [0] * self.num_train_processes

        if self.map == "V1" or self.map == "V3":
            self.eval_mazes = [0]
        elif self.map == "V2" or self.map == "V4":
            self.eval_mazes = [0, 1]

        self.num_test_processes = len(self.eval_mazes)

        self.n_eval = 500
        self.gpu_ids_train = [0, 1]
        self.gpu_ids_test = [0, 1]
        self.seed = random.randint(0, 10000)

        # Model/optimizer hyperparameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.lr = 7e-5
        self.tau = 1.0
        self.clip_grad_norm = 10.0
        self.value_loss_coef = 0.5
        self.amsgrad = True
        self.goal_coef = 0.5
        self.goal_batch_size = 50
        self.minimum_warmup = self.num_train_processes * 100
        self.weight_decay = 0

        # Gym environment settings
        self.scaled_resolution = (42, 42)
        self.living_reward = -0.0025   # 4-frame stack, so living reward is quadrupled
        self.target_reward = 10.0
        self.non_target_penalty = 1.0
        self.non_target_break = True
        self.timeout_penalty = 0.1

        # Logging-related
        self.mazes_path_root = './maps/{}/'.format(self.map)
        now = datetime.datetime.now()
        nowDate = now.strftime('%Y-%m-%d-%H:%M:%S')
        self.log_file = nowDate
        if not os.path.exists('./wgt'):
            os.mkdir('./wgt')
        self.weight_dir = './wgt/{}_wgt/'.format(nowDate)


params = Params()

def log_params(is_gen_maps=False):
    if is_gen_maps:
        path = './log/{}'.format(params.gen_maps_log_file)
    else:
        path = './log/{}'.format(params.log_file)

    msg = str('start time\t{}\n'.format(time.strftime('%X %x %Z')))

    params_dict = params.__dict__
    for key in params_dict.keys():
        msg += '{}\t{}\n'.format(key, str(params_dict[key]))

    msg += '\n' + '\t'.join(['time', 'numUpdates', 'mazeId', 'saveModelIdx', 'avgReward', 'avgLength', 'successRate', 'bestRate']) + '\n'
    csv_path = path + '.csv'
    if not os.path.isdir('./log'):
        os.mkdir('./log')

    with open(csv_path, 'w') as file:
        file.write(msg)

