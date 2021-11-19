# Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning (NeurIPS'21) [[arxiv](http://arxiv.org/abs/2110.12985)]

This repository contains sourcecode of our paper "Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning" at NeurIPS 2021.

This code contains our method, GACE&GDAN, for Visual Navigation tasks.



## Dependencies

- python3
- pytorch 1.7 +
- tensorboard 2.4
- MazeExplorer [[link](https://github.com/microsoft/MazeExplorer#installation)]


## Run

Before you run this script, you have to check `params.py` for allocating your hardward settings properly.

`self.gpu_ids_train = [0,1]` 

and 

`self.gpu_ids_test = [0,1]`

are your id of gpus. If you have only one, leave [0] and if more, you may allocate more gpus.

Especially, `self.num_training_process` is more than yours, it cause an error.



When you are ready, run the script below:

`python main.py`



## Citation

If you think our research is helpful, please consider citing,
```
@inproceedings{kim2021goal,
  title={Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning},
  author={Kim, Kibeom and Lee, Min Whoo and Kim, Yoonsung and Ryu, JeHwan and Lee, Minsu and Zhang, Byoung-Tak},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
