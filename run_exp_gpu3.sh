#!/bin/bash

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=3
#numactl -C 15 -m 1 python -u main.py --env hopper-medium-expert-v2 --seed 0 &> hme_seed_0
#numactl -C 15 -m 1 python -u main.py --env hopper-medium-expert-v2 --seed 1 &> hme_seed_1
#numactl -C 15 -m 1 python -u main.py --env hopper-medium-expert-v2 --seed 2 &> hme_seed_2
#numactl -C 15 -m 1 python -u main.py --env hopper-expert-v2 --seed 2 &> he_seed_2

#numactl -C 15 -m 1 python -u main_walker.py --env walker2d-medium-expert-v2 --seed 0 &> wme_seed_0
#numactl -C 15 -m 1 python -u main_walker.py --env walker2d-medium-expert-v2 --seed 1 &> wme_seed_1
#numactl -C 15 -m 1 python -u main_walker.py --env walker2d-medium-expert-v2 --seed 2 &> wme_seed_2
#numactl -C 15 -m 1 python -u main_walker.py --env walker2d-expert-v2 --seed 2 &> we_seed_2
