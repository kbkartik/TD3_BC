#!/bin/bash

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=2
#numactl -C 12 -m 1 python -u main.py --env hopper-medium-replay-v2 --seed 0 &> hmr_seed_0
#numactl -C 12 -m 1 python -u main.py --env hopper-medium-replay-v2 --seed 1 &> hmr_seed_1
#numactl -C 12 -m 1 python -u main.py --env hopper-medium-replay-v2 --seed 2 &> hmr_seed_2
#numactl -C 12 -m 1 python -u main.py --env hopper-expert-v2 --seed 1 &> he_seed_1

#numactl -C 12 -m 1 python -u main_walker.py --env walker2d-medium-replay-v2 --seed 0 &> wmr_seed_0
#numactl -C 12 -m 1 python -u main_walker.py --env walker2d-medium-replay-v2 --seed 1 &> wmr_seed_1
#numactl -C 12 -m 1 python -u main_walker.py --env walker2d-medium-replay-v2 --seed 2 &> wmr_seed_2
#numactl -C 12 -m 1 python -u main_walker.py --env walker2d-expert-v2 --seed 1 &> we_seed_1
