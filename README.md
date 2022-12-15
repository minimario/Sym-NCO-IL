# Sym-NCO-IL: Imitation Learning with Symmetric Learning for Combinatorial Optimization. 

The Sym-NCO ([Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization](https://openreview.net/forum?id=kHrE2vi5Rvs)) was deep reinforcement learning-based neural combinatorial optimization scheme by exploiting symmetric nature of combinatorial optimization, published at NeurIPS 2022. 

Sym-NCO-IL is extension of Sym-NCO's motivation into imitation learning with sparse labeled data setting. 

We think symmetric learning scheme can gives clear gain in sparse labeled data setting; therefore IL is more clear benchmark than DRL for making furtherwork of Sym-NCO. 

This repository is for people who want to research symmetric learning for combinatorial optimization; feel free to use this code for your further research.


## Source code implementation

This code is originally implemented based on  [Attention Model](https://github.com/wouterkool/attention-learn-to-route) , which is source code of the paper   [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm) which has been accepted at [ICLR 2019](https://iclr.cc/Conferences/2019), cite as follows:

```
@inproceedings{
    kool2018attention,
    title={Attention, Learn to Solve Routing Problems!},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxBFsRqYm},
}
```




## Dependencies 

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)


## How to generate expert labeled data?

We used LKH3 as expert, but feel free to use any expert if you want. We provide expert data at "expert_data/"

## Quick Start

You can run quick start with: 

```bash
python run.py --label_aug --consistancy_learning
```

## More options

**AM with IL (baseline without symmetricity)** 

```bash
python run.py
```

**AM with symmetric data augmentation**

```bash
python run.py --label_aug
```

**AM with unsupervised symmetric consistancy learning**

```bash
python run.py --consistancy_learning
```

**AM with Semi-supervised symmetric learning**

```bash
python run.py --label_aug --consistancy_learning
```

**Change expert data path**

```bash
--datapath expert_data/[DATA NAME].pkl
```

**Change Problem Size**

```bash
--graph_size 20
```
---
## Important Remark

* Read code for AM first!! [Attention Model](https://github.com/wouterkool/attention-learn-to-route)
* Only code for TSP is implemented, but other problems can be easily implemented.
* I did not implement "eval.py". I just check performances via validation score at "run.py". However, it is recommented to modify "eval.py".

## Reference

* If you want to make your paper based on this code, please consider to refer:



**AM with Imitation Learning for Hardware Routing**
```
@inproceedings{kim2021imitation,
  title={Imitation Learning for Simultaneous Escape Routing},
  author={Kim, Minsu and Park, Hyunwook and Son, Keeyoung and Kim, Seongguk and Kim, Haeyeon and Kim, Jihun and Song, Jinwook and Ku, Youngmin and Park, Jounggyu and Kim, Joungho},
  booktitle={2021 IEEE 30th Conference on Electrical Performance of Electronic Packaging and Systems (EPEPS)},
  pages={1--3},
  year={2021},
  organization={IEEE}
}
```

**Sym-NCO**
```
@article{kim2022sym,
  title={Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization},
  author={Kim, Minsu and Park, Junyoung and Park, Jinkyoo},
  journal={arXiv preprint arXiv:2205.13209},
  year={2022}
}
```

**AM-IL with placement symmetricity for multi-task hardware placement**
```
@inproceedings{kimcollaborative,
  title={Collaborative symmetricity exploitation for offline learning of hardware design solver},
  author={Kim, Haeyeon and Kim, Minsu and Kim, Joungho and Park, Jinkyoo},
  booktitle={3rd Offline RL Workshop: Offline RL as a''Launchpad''}
}
```
