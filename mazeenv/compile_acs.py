# Modified from https://github.com/microsoft/MazeExplorer/blob/e66a2d405e08bc75e51bd38a2b96959c554fe773/mazeexplorer/compile_acs.py

# This script uses acc (from https://github.com/rheit/acc) to compile the acs scripts.
import os
import subprocess
import mazeexplorer


dir_path = os.path.dirname(os.path.realpath(__file__))


def compile_acs(mazes_path):
    os.makedirs(os.path.join(mazes_path, "outputs", "sources"))
    os.makedirs(os.path.join(mazes_path, "outputs", "images"))

    mazeexplorer_path = os.path.dirname(os.path.realpath(mazeexplorer.__file__))
    acc_path = os.path.join(mazeexplorer_path, "acc/acc")

    # if not os.path.isfile(acc_path):
    #     print("Compiling ACC as File not does exist: ", acc_path, "")
    #     subprocess.call(["make", "-C", os.path.join(dir_path, "acc")])

    maze_acs_path = os.path.join(dir_path, "maze.acs")
    output_file_path = os.path.join(dir_path, '..', mazes_path, "outputs", "maze.o")
    subprocess.call([acc_path, "-i", "./acc", maze_acs_path, output_file_path])