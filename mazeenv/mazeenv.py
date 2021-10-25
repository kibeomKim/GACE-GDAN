# Modified from https://github.com/microsoft/MazeExplorer/blob/e66a2d405e08bc75e51bd38a2b96959c554fe773/mazeexplorer/mazeexplorer.py

from pathlib import Path

from .vizdoom_gym import VizDoom

class MazeEnv(VizDoom):
    @staticmethod
    def load_vizdoom_env(mazes_path, scaled_resolution=(42, 42),
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
        cfg_paths = list(Path(mazes_path).glob("*.cfg"))
        if len(cfg_paths) != 1:
            raise ValueError("Invalid number of cfgs within the mazes path: ", len(cfg_paths))
        return VizDoom(cfg_paths[0], scaled_resolution=scaled_resolution, living_reward=living_reward,
            target_reward=target_reward, non_target_penalty=non_target_penalty,
            timeout_penalty=timeout_penalty, non_target_break=non_target_break)
