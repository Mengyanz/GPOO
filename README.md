# Supplementary Code for Gaussian Process Bandits with Aggregated Feedback

We provide the implementation for our algorithm Gaussian Process Optimistic Optimisation (GPOO) and related algorithms in *supplementary_code*. 
Full paper (including appendix) can be found [here](https://arxiv.org/abs/2112.13029).

## Usage

In terminal:
```
conda env create -f environment.yml
conda activate gpoo_env
python run_sim.py -h
```
This will gives 
```
Run Simulation for GPOO project.

positional arguments:
  opt_num               choose what f to use, choices: 1,2,3

optional arguments:
  -h, --help            show this help message and exit
  --n N                 budget (should be positive integer)
  --r R                 number of repeat (should be positive integer)
  --alg [ALG [ALG ...]]
                        please list all algorithms to run. Choices: StoOO, GPOO, GPTree, SK
```

Here is an example of usage:
to run GPOO algorithm with function choice 1, with budget 80, 30 independent runs: 
```
python run_sim.py 1 --n 80 --r 30 --alg GPOO
```

Cite us by
```
@article{zhang2022GPOO,
  title={Gaussian Process Bandits with Aggregated Feedback},
  author={Zhang, Mengyan and Tsuchida, Russell and Ong, Cheng Soon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
