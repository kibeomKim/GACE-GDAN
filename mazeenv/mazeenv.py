# Modified from https://github.com/microsoft/MazeExplorer/blob/e66a2d405e08bc75e51bd38a2b96959c554fe773/mazeexplorer/mazeexplorer.py

import datetime
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import cv2

from .maze import generate_mazes
from .script_manipulator import write_config, write_acs
from .vizdoom_gym import VizDoom
from mazeexplorer.wad import generate_wads
from .compile_acs import compile_acs

dir_path = os.path.dirname(os.path.realpath(__file__))


class MazeEnv(VizDoom):
    def __init__(self, unique_maps=False, number_maps=1, size=(10, 10), random_spawn=False, random_textures=False,
                 random_key_positions=False, seed=None, clip=(-1, 1),
                 floor_texture="CEIL5_2", ceiling_texture="CEIL5_1", wall_texture="STONE2",
                 action_frame_repeat=4, actions="MOVE_FORWARD TURN_LEFT TURN_RIGHT", scaled_resolution=(42, 42),
                 episode_timeout=1500, complexity=.7, density=.7, data_augmentation=False, mazes_path=None,
                 key_categories=None, random_key_textures=False, living_reward=-0.01,
                 target_reward=1.0, non_target_penalty=1.0, timeout_penalty=1.0,
                 maze_layout=None, default_spawn_pos=None, default_key_pos=None,
                 resolution="RES_160X120", non_target_break=False,
                 acs_path=None, texture_list=None, keys_used_list=None,
                 use_key_boxes=False, shuffle_obj_pos=False, boxes_dims=(1, 1),
                 random_player_spawn_angle=True, def_player_spawn_angle=0.0,
                 gen_map=False):

        """
        Gym environment where the goal is to collect a preset number of keys within a procedurally generated maze.
        MazeExplorer is a customisable 3D benchmark for assessing generalisation in Reinforcement Learning.
        :params unique_maps: if set, every map will only be seen once. cfg files will be recreated after all its maps have been seen.
        :param number_maps: number of maps which are contained within the cfg file. If unique maps is set, this acts like a cache of maps
        :param size: the size of generated mazes in the format (width, height)
        :param random_spawn: whether to randomise the spawn each time the environment is reset
        :param random_textures: whether to randomise the textures for each map each time the environment is reset
        :param random_key_positions: whether to randomise the position of the keys each time the environment is reset
        :param seed: seed for random, used to determine the other that the doom maps should be shown.
        :param clip: how much the reward returned on each step should be clipped to
        :param floor_texture: the texture to use for the floor, options are in mazeexplorer/content/doom_textures.acs
        Only used when random_textures=False
        :param ceiling_texture: the texture to use for the ceiling, options are in mazeexplorer/content/doom_textures.acs
        Only used when random_textures=False
        :param wall_texture: the texture to use for the walls, options are in mazeexplorer/content/doom_textures.acs
        Only used when random_textures=False
        :param action_frame_repeat: how many game tics should an action be active
        :param actions: the actions which can be performed by the agent
        :param scaled_resolution: resolution (height, width) of the observation to be returned with each step
        :param episode_timeout: the number of ticks in the environment before it time's out
        :param complexity: float between 0 and 1 describing the complexity of the generated mazes
        :param density: float between 0 and 1 describing the density of the generated mazes
        :param data_augmentation: bool to determine whether or not to use data augmentation
            (adding randomly colored, randomly sized boxes to observation)
        :type mazes_path: path to where to save the mazes
        """
        self.unique_maps = unique_maps
        self.number_maps = number_maps
        self.size = size
        self.random_spawn = random_spawn
        self.random_textures = random_textures
        self.random_key_positions = random_key_positions
        self.seed = seed
        self.clip = clip
        self.actions = actions
        self.mazes = None
        self.action_frame_repeat = action_frame_repeat
        self.scaled_resolution = scaled_resolution
        self.episode_timeout = episode_timeout
        self.complexity = complexity
        self.density = density
        self.data_augmentation = data_augmentation
        self.key_categories = key_categories
        if self.key_categories is None:
            raise ValueError('Must provide key texture categories')
        self.random_key_textures = random_key_textures

        # The mazeexplorer textures to use if random textures is set to False
        self.wall_texture = wall_texture
        self.floor_texture = floor_texture
        self.ceiling_texture = ceiling_texture

        self.living_reward = living_reward
        self.target_reward = target_reward
        self.non_target_penalty = non_target_penalty
        self.timeout_penalty = timeout_penalty
        self.maze_layout = maze_layout
        self.default_spawn_pos = default_spawn_pos
        self.default_key_pos = default_key_pos
        self.resolution = resolution
        self.non_target_break = non_target_break

        self.use_key_boxes = use_key_boxes
        self.boxes_dims = boxes_dims
        self.shuffle_obj_pos = shuffle_obj_pos
        self.random_player_spawn_angle = random_player_spawn_angle
        self.def_player_spawn_angle = def_player_spawn_angle

        self.gen_map = gen_map

        self.acs_path = acs_path
        if texture_list is None:
            raise ValueError('Must provide texture list')
        self.texture_list = texture_list

        self.keys_used_list = keys_used_list
        if keys_used_list is None:
            self.keys_used_list = list(range(len(self.key_categories)))

        self.mazes_path = mazes_path if mazes_path is not None else tempfile.mkdtemp()
        # create new maps and corresponding config
        shutil.rmtree(self.mazes_path, ignore_errors=True)
        os.mkdir(self.mazes_path)

        self.cfg_path = self.generate_mazes()

        # start map with -1 since it will always be reseted one time.
        self.current_map = -1

        super().__init__(self.cfg_path, number_maps=self.number_maps, scaled_resolution=self.scaled_resolution,
                         action_frame_repeat=self.action_frame_repeat, seed=seed,
                         data_augmentation=self.data_augmentation, clip=self.clip,
                         living_reward=self.living_reward, target_reward=self.target_reward,
                         non_target_penalty=self.non_target_penalty,
                         timeout_penalty=self.timeout_penalty, non_target_break=self.non_target_break,
                         num_obj_to_spawn=len(self.keys_used_list),
                         gen_map=self.gen_map)


    def generate_mazes(self):
        """
        Generate the maze cfgs and wads and place them in self.mazes_path
        :return: path to the maze_cfg
        """
        # edit base acs template to reflect user specification
        write_acs(random_player_spawn=self.random_spawn,
                  random_textures=self.random_textures,
                  random_key_positions=self.random_key_positions,
                  map_size=self.size,
                  number_maps=self.number_maps,
                  floor_texture=self.floor_texture,
                  ceiling_texture=self.ceiling_texture,
                  wall_texture=self.wall_texture,
                  key_categories=self.key_categories,
                  random_key_textures=self.random_key_textures,
                  seed=self.seed,
                  default_spawn_pos=self.default_spawn_pos,
                  default_key_pos=self.default_key_pos,
                  texture_list=self.texture_list,
                  keys_used_list=self.keys_used_list,
                  acs_path=self.acs_path,
                  use_key_boxes=self.use_key_boxes,
                  boxes_dims=self.boxes_dims,
                  shuffle_obj_pos=self.shuffle_obj_pos,
                  random_player_spawn_angle=self.random_player_spawn_angle,
                  def_player_spawn_angle=self.def_player_spawn_angle)

        compile_acs(self.mazes_path)

        # generate .txt maze files
        generate_mazes(maze_id=self.mazes_path + "/" + str(self.size[0]) + "x" \
                               + str(self.size[1]),
                       num=self.number_maps,
                       rows=self.size[0], columns=self.size[1],
                       seed=self.seed, complexity=self.complexity, density=self.density,
                       maze_layout=self.maze_layout)

        outputs = os.path.join(self.mazes_path, "outputs/")

        # convert .txt mazes to wads and link acs scripts
        try:
            generate_wads(self.mazes_path + "/" + str(self.size[0]) + "x" + str(self.size[1]),
                          self.mazes_path + "/" + str(self.size[0]) + "x" + str(self.size[1]) + ".wad",
                          outputs + "maze.o")
        except FileNotFoundError as e:
            raise FileNotFoundError(e.strerror + "\n"
                                                 "Have you pulled the required submodules?\n"
                                                 "If not, use the line:\n\n\t"
                                                 "git submodule update --init --recursive")
        cfg = write_config(self.mazes_path + "/" + str(self.size[0]) + "x" + str(self.size[1]),
                           self.actions, episode_timeout=self.episode_timeout,
                           living_reward=self.living_reward, resolution=self.resolution)

        return cfg

    def reset(self):
        """Resets environment to start a new mission.
        If `unique_maps` is set and and all cached maps have been seen, it wil also generate
        new maps using the ACC script. Otherwise if there is more than one maze 
        it will randomly select a new maze for the list.
        :return: initial observation of the environment as an rgb array in the format (rows, columns, channels) """
        if not self.unique_maps:
            return super().reset()
        else:
            self.current_map += 1
            if self.current_map > self.number_maps:
                print("Generating new maps")

                if self.seed is not None:
                    np.random.seed(self.seed)

                self.seed = np.random.randint(np.iinfo(np.int32).max)

                # create new maps and corresponding config
                shutil.rmtree(self.mazes_path)

                os.mkdir(self.mazes_path)
                self.cfg_path = self.generate_mazes()

                # reset the underlying DoomGame class
                self.env.load_config(self.cfg_path)
                self.env.init()
                self.current_map = 0

            self.doom_map = "map" + str(self.current_map).zfill(2)
            self.env.set_doom_map(self.doom_map)
            self.env.new_episode()
            self._rgb_array = self.env.get_state().screen_buffer
            observation = self._process_image()
            return observation

    def save(self, destination_dir):
        """
        Save the maze files to a directory
        :param destination_dir: the path of where to save the maze files
        """
        shutil.copytree(self.mazes_path, destination_dir)

    @staticmethod
    def load_vizdoom_env(mazes_path, number_maps, num_obj_to_spawn, action_frame_repeat=4, scaled_resolution=(42, 42),
                         living_reward=-0.0025, target_reward=5.0, non_target_penalty=1.0,
                         timeout_penalty=1.0, non_target_break=False):
        """
        Takes the path to a maze cfg or a folder of mazes created by mazeexplorer.save() and returns a vizdoom environment
        using those mazes
        :param mazes_path: path to a .cfg file or a folder containg the cfg file
        :param number_maps: number of maps contained within the wad file
        :param action_frame_repeat: how many game tics should an action be active
        :param scaled_resolution: resolution (height, width) of the observation to be returned with each step
        :return: VizDoom gym env
        """

        if str(mazes_path).endswith(".cfg"):
            return VizDoom(mazes_path, number_maps=number_maps, scaled_resolution=scaled_resolution,
                           action_frame_repeat=action_frame_repeat, living_reward=living_reward,
                           target_reward=target_reward, non_target_penalty=non_target_penalty,
                           timeout_penalty=timeout_penalty, non_target_break=non_target_break,
                           num_obj_to_spawn=num_obj_to_spawn)
        else:
            cfg_paths = list(Path(mazes_path).glob("*.cfg"))
            if len(cfg_paths) != 1:
                raise ValueError("Invalid number of cfgs within the mazes path: ", len(cfg_paths))
            return VizDoom(cfg_paths[0], number_maps=number_maps, scaled_resolution=scaled_resolution,
                           action_frame_repeat=action_frame_repeat, living_reward=living_reward,
                           target_reward=target_reward, non_target_penalty=non_target_penalty,
                           timeout_penalty=timeout_penalty, non_target_break=non_target_break,
                           num_obj_to_spawn=num_obj_to_spawn)

    @staticmethod
    def generate_video(images_path, movie_path):
        """
        Generates a video of the agent from the images saved using record_path
        Example:
        ```python
        images_path = "path/to/save_images_dir"
        movie_path = "path/to/movie.ai"
        env = MazeNavigator(record_path=images_path)
        env.reset()
        for _ in range(100):
            env.step_record(env.action_space.sample(), record_path=images_path)
        MazeNavigator.generate_video(images_path, movie_path)
        ```
        :param images_path: path of the folder containg the generated images
        :param movie_path: file path ending with .avi to where the movie should be outputted to.
        """

        if not movie_path.endswith(".avi"):
            raise ValueError("movie_path must end with .avi")

        images = sorted([img for img in os.listdir(images_path) if img.endswith(".png")])

        if not len(images):
            raise FileNotFoundError("Not png images found within the images path")

        frame = cv2.imread(os.path.join(images_path, images[0]))
        height, width, _ = frame.shape

        video = cv2.VideoWriter(movie_path, 0, 30, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(images_path, image)))

        cv2.destroyAllWindows()
        video.release()
