import vizdoom as vzd
import itertools as it
from random import choice
import cv2
import gym

import pdb


def initialize_vzd(config_file_path):
    print("Initializing doom...")

    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)   # for learning, it will be False
    game.set_render_weapon(False)
    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of he current episode/level .
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.init()
    print("Doom initialized.")
    return game

def initialize_vzd_visible(config_file_path):
    print("Initializing doom...")

    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(True)   # for learning, it will be False
    game.set_render_weapon(False)
    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of he current episode/level .
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.init()
    print("Doom initialized.")
    return game

def get_max_actions(config_file_path):
    game = initialize_vzd(config_file_path)
    n_button = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n_button)]

    game.close()
    return len(actions)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)