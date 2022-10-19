#!/bin/bash

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=0
#numactl -C 2 -m 0 python -u main.py --env hopper-random-v2 --seed 0 &> hr_seed_0
#numactl -C 2 -m 0 python -u main.py --env hopper-random-v2 --seed 1 &> hr_seed_1
#numactl -C 2 -m 0 python -u main.py --env hopper-random-v2 --seed 2 &> hr_seed_2

numactl -C 2 -m 0 python -u main_walker.py --env walker2d-random-v2 --seed 0 &> wr_seed_0
numactl -C 2 -m 0 python -u main_walker.py --env walker2d-random-v2 --seed 1 &> wr_seed_1
numactl -C 2 -m 0 python -u main_walker.py --env walker2d-random-v2 --seed 2 &> wr_seed_2