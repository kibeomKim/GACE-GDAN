from collections import OrderedDict
import time
import os
import random

class Params:
    def __init__(self):

        date_str = '20211022_code_test'
        map_date_str = 'V1'   # '20201208_L3_large_new'  # '20201112_L0_77' # '20201130_L2_large'   # '20201123_L1_77_new'
        self.map = map_date_str
        # Configs for running trains/test
        self.num_train_processes = 20
        self.num_test_processes = 2
        self.n_random_trial = 4
        self.n_eval = 500
        self.num_steps_bw_updates = 50
        self.gpu_ids_train = [0, 1]
        self.gpu_ids_test = [0, 1]
        self.seed = random.randint(0,10000)  # 0

        self.train_mazes = [0] * 20
        # self.eval_mazes = list(range(10, 13))
        self.eval_mazes = [0]
        self.mazes_path_root = './maps/{}/'.format(map_date_str)
        self.num_test_random_mazes = 1
        # self.mazes_path_root = './maps/maps_0302_8x8rooms_step300/test/'
        # self.num_test_random_mazes = 3

        # Logging-related
        self.log_file = date_str
        if not os.path.exists('./wgt'):
            os.mkdir('./wgt')
        self.weight_dir = './wgt/{}_wgt/'.format(date_str)
        self.gen_maps_log_file = '{}_gen_map'.format(map_date_str)
        self.log_debug = False

        # Model/optimizer hyperparameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.lr = 1e-5  # 7e-5
        self.tau = 1.0
        self.clip_grad_norm = 10.0
        self.value_loss_coef = 0.5
        self.amsgrad = True
        self.goal_coef = 0.5
        self.goal_batch_size = 50
        self.minimum_warmup = self.num_train_processes * 100
        self.weight_decay = 0  # 0.00005   ## no touch

        # Configuration for generating maps
        self.num_maps = 2                       # set to 1 for test, 10 for train
        self.gen_map_seed = 0                      # set to 10 for test, 0 for train
        self.episode_timeout = 400  # original
        # self.episode_timeout = 500000 #DEBUG
        self.resolution = 'RES_160X120'
        # self.resolution = 'RES_1280X1024'  # DEBUG
        self.acs_script = 'mazeenv_acs_template.txt'
        self.gen_maps_dir_prefix = map_date_str

        self.random_player_spawn = False              # turn off during test
        # self.def_player_spawn_pos = None
        self.def_player_spawn_pos = (420.0, 420.0)
        self.random_player_spawn_angle = False   # turn off during test

        self.def_player_spawn_angle = None

        self.random_wall_textures = False  # Turn off during test
        # all texture (ME)
        self.texture_list = ["ASHWALL2", "ASHWALL3", "ASHWALL4", "ASHWALL6", "ASHWALL7", "BFALL1", "BFALL2", "BFALL3", "BFALL4", "BIGBRIK1", "BIGBRIK2", "BIGBRIK3", "BIGDOOR2", "BIGDOOR3", "BIGDOOR4", "BIGDOOR5", "BLAKWAL1", "BLAKWAL2", "BRICK1", "BRICK2", "BRICK3", "BRICK4", "BRICK5", "BRICK6", "BRICK7", "BRICK8", "BRICK9", "BRICK10", "BRICK11", "BRICK12", "BRICKLIT", "BRONZE1", "BRONZE2", "BRONZE3", "BRONZE4", "BROVINE2", "BROWN1", "BROWN144", "BROWN96", "BROWNGRN", "BROWNHUG", "BROWNPIP", "BRWINDOW", "BSTONE1", "BSTONE2", "BSTONE3", "CEMENT1", "CEMENT2", "CEMENT3", "CEMENT4", "CEMENT5", "CEMENT6", "CEMENT7", "CEMENT9", "COMPBLUE", "COMPSPAN", "COMPSTA1", "COMPSTA2", "COMPTALL", "COMPWERD", "CRACKLE2", "CRACKLE4", "CRATE1", "CRATE2", "CRATE3", "CRATELIT", "CRATWIDE", "DOORBLU", "DOORRED", "DOORSTOP", "DOORTRAK", "DOORYEL", "FIREWALA", "FIREWALB", "FIREWALL", "GRAY1", "GRAY4", "GRAY5", "GRAYBIG", "GRAYVINE", "GSTONE1", "GSTONE2", "GSTVINE1", "GSTVINE2", "ICKWALL1", "ICKWALL2", "ICKWALL3", "LITE3", "LITE5", "LITEBLU1", "LITEBLU4", "MARBGRAY", "MARBLE1", "MARBLE2", "MARBLE3", "MARBLOD1", "METAL", "METAL1", "METAL2", "METAL3", "METAL4", "METAL5", "METAL6", "METAL7", "MODWALL1", "MODWALL2", "MODWALL4", "NUKEDGE1", "PANBOOK", "PANBORD1", "PANBORD2", "PANCASE1", "PANCASE2", "PANEL1", "PANEL2", "PANEL4", "PANEL5", "PANEL6", "PANEL7", "PANEL8", "PANEL9", "PIPE1", "PIPE2", "PIPE4", "PIPE6", "PIPEWAL1", "PIPEWAL2", "PLAT1", "REDWALL", "ROCK1", "ROCK2", "ROCK3", "ROCK4", "ROCK5", "ROCKRED1", "ROCKRED2", "SFALL1", "SFALL2", "SFALL3", "SFALL4", "SHAWN2", "SILVER1", "SILVER2", "SILVER3", "SK_LEFT", "SK_RIGHT", "SKIN2", "SLADWALL", "SP_HOT1", "SPACEW2", "SPACEW3", "SPACEW4", "SPCDOOR1", "SPCDOOR2", "SPCDOOR3", "SPCDOOR4", "STARBR2", "STARG1", "STARG2", "STARG3", "STARGR1", "STARGR2", "STARTAN2", "STARTAN3", "STONE", "STONE2", "STONE3", "STONE4", "STONE5", "STONE6", "STONE7", "STUCCO", "STUCCO1", "SUPPORT2", "SUPPORT3", "TANROCK2", "TANROCK3", "TANROCK4", "TANROCK5", "TANROCK7", "TANROCK8", "TEKBRON1", "TEKBRON2", "TEKGREN1", "TEKGREN2", "TEKGREN3", "TEKGREN4", "TEKGREN5", "TEKLITE", "TEKLITE2", "TEKWALL1", "TEKWALL4", "TEKWALL6", "WOOD1", "WOOD3", "WOOD5", "WOOD6", "WOOD7", "WOOD8", "WOOD9", "WOOD12", "WOODMET1", "WOODVERT", "ZDOORB1", "ZDOORF1", "ZELDOOR", "ZIMMER2", "ZIMMER5", "ZIMMER7", "ZIMMER8", "ZZWOLF1", "ZZWOLF5", "ZZWOLF9", "ZZWOLF10", "ZZWOLF11", "ZZZFACE6", "ZZZFACE7", "ZZZFACE8", "ZZZFACE9"]
        # 10 textures (OOME)
        # self.texture_list = ['PIPE6', 'BRICK4', 'PANEL1', 'STARBR2', 'ZZZFACE9', 'SPCDOOR4', 'STARTAN2', 'TEKGREN4', 'BIGDOOR3', 'ICKWALL1']

        self.random_key_positions = True   # turn off during test
        self.def_key_pos = None
        self.shuffle_obj_pos = True
        self.random_key_textures = True    # turn off during test

        # ME
        # self.key_categories = OrderedDict({'Card': ['RedCard']})
        # self.keys_used_list = [0]
        # OOME
        self.key_categories = OrderedDict({
            'Card': ['RedCard', 'BlueCard'],
            'Armor': ['RedArmor', 'GreenArmor'],
            'Skull': ['YellowSkull', 'BlueSkull'],  # , 'RedSkull'

            # 'Gun': ['Shotgun', 'Chaingun'],
            'Bonus': ['HealthBonus', 'ArmorBonus'],
        })
        self.keys_used_list = [0, 1, 2, 3]

        # DEBUG
        # self.key_categories = OrderedDict({
        #     'Card': ['RedCard'],
        #     'Armor': ['RedArmor'],
        # })
        # self.keys_used_list = [0, 1]

        self.map_size = (10, 10)
        self.complexity = 0.0   # 0.7
        self.density = 0.0  # 0.7
        self.maze_layout = None
        self.use_key_boxes = True    # True for OOME, False for ME)
        self.boxes_dims = (2, 2)  # num of boxes must be >= num of objs

        # Gym environment settings
        self.reward_clip = (-10.0, 10.0)
        self.action_frame_repeat = 4
        self.scaled_resolution = (42, 42)
        self.data_augmentation = False
        self.living_reward = -0.0025   # 4-frame stack, so living reward is quadrupled
        self.target_reward = 10.0
        self.non_target_penalty = 1.0
        self.non_target_break = True
        self.timeout_penalty = 0.1


params = Params()
debug_log_path = './log/' + params.log_file + '.log'


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
    # commented out as DEBUG
    # if os.path.isfile(csv_path):
    #     raise ValueError('Log CSV file already exists')
    with open(csv_path, 'w') as file:
        file.write(msg)

    if params.log_debug:
        msg = str('start time\t{}\n'.format(time.strftime('%X %x %Z')))
        msg += '\n' + '\t'.join(['rank', 'episode', 'maze_id', 'total_rew', 'step', 'good', \
            'n_update', 'action', 'value', 'logit', 'prob', 'rew', 'done', 'info']) + '\n'
        with open(path + '.log', 'w') as file:
            file.write(msg)
