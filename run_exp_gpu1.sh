#!/bin/bash

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=1
#numactl -C 4 -m 0 python -u main.py --env hopper-medium-v2 --seed 0 &> hm_seed_0
#numactl -C 4 -m 0 python -u main.py --env hopper-medium-v2 --seed 1 &> hm_seed_1
#numactl -C 4 -m 0 python -u main.py --env hopper-medium-v2 --seed 2 &> hm_seed_2
#numactl -C 4 -m 0 python -u main.py --env hopper-expert-v2 --seed 0 &> he_seed_0

#numactl -C 4 -m 0 python -u main_walker.py --env walker2d-medium-v2 --seed 0 &> wm_seed_0
#numactl -C 4 -m 0 python -u main_walker.py --env walker2d-medium-v2 --seed 1 &> wm_seed_1
#numactl -C 4 -m 0 python -u main_walker.py --env walker2d-medium-v2 --seed 2 &> wm_seed_2
#numactl -C 4 -m 0 python -u main_walker.py --env walker2d-expert-v2 --seed 0 &> we_seed_0
