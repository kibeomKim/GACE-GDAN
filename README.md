# Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning (NeurIPS'21) [[arxiv](http://arxiv.org/abs/2110.12985)]

This repository contains sourcecode of our paper "Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning" at NeurIPS 2021.

This code contains our method, GACE&GDAN, for Visual Navigation tasks.



## Dependencies

- python3
- pytorch 1.7 +
- tensorboard 2.4
- numpy 1.17
- setproctitle 1.2
- Multi-target Visual Navigation environments ([link](https://github.com/lionminhu/multitarget-visnav))


## Run

Before you run this script, please check `params.py` for allocating your GPU properly. Particularly, referring to below parameters,

```
self.gpu_ids_train = [0,1]
```
and
```
self.gpu_ids_test = [0,1]
```

indicate which GPUs to allocate for training and evaluating, respectively. If you have only one, set these parameters to [0]. Otherwise, you may allocate more GPUs.

Please make sure that `self.num_training_process` is set according to the number of CPU cores and the amount of GPU memory.


When you are ready, run the script to start the training:
```
python main.py
```



## Citation

If you find our research helpful, please consider citing our paper,
```
@inproceedings{kim2021goal,
  title={Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning},
  author={Kim, Kibeom and Lee, Min Whoo and Kim, Yoonsung and Ryu, JeHwan and Lee, Minsu and Zhang, Byoung-Tak},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
