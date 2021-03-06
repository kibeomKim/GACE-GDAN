import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp
import torch.optim as optim

import argparse

from multiprocessing import Manager
from multiprocessing.managers import BaseManager

from params import params, log_params

from GACE_GDAN.train import run_sim
from GACE_GDAN.eval import test
from GACE_GDAN.goal_storage import goal_storage
from GACE_GDAN.models import GDAN

from utils.shared_optim import SharedAdam


def main():
    log_params()

    mp.set_start_method('spawn')
    count = mp.Value('i', 0)
    lock = mp.Lock()

    shared_model = GDAN()
    shared_model = shared_model.share_memory()

    shared_optimizer = SharedAdam(shared_model.parameters(), lr=params.lr, amsgrad=params.amsgrad,
                                  weight_decay=params.weight_decay)
    shared_optimizer.share_memory()

    BaseManager.register('goal_storage', goal_storage)
    manager = BaseManager()
    manager.start()
    shared_storage = manager.goal_storage()

    processes = []
    for rank in range(params.num_train_processes):
        p = mp.Process(target=run_sim, args=(rank, shared_model, shared_optimizer, count, lock, shared_storage))
        p.start()
        processes.append(p)

    for rank in range(params.num_test_processes):
        p = mp.Process(target=test, args=(rank, shared_model, shared_optimizer, count, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
