#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python mbpo.py --env "Hopper-v2"      --rollout_length 1 --method "tom" --seed 1 --no_disc  --f "chi"  &
CUDA_VISIBLE_DEVICES=1 python mbpo.py --env "Walker2d-v2"    --rollout_length 1 --method "tom" --seed 1 --no_disc  --f "chi"  &
CUDA_VISIBLE_DEVICES=1 python mbpo.py --env "HalfCheetah-v2" --rollout_length 1 --method "tom" --seed 1 --no_disc  --f "chi"  &
CUDA_VISIBLE_DEVICES=1 python mbpo.py --env "Humanoid-v2"    --rollout_length 1 --method "tom" --seed 1 --no_disc  --f "chi"  &
CUDA_VISIBLE_DEVICES=0 python mbpo.py --env "Ant-v2"         --rollout_length 1 --method "tom" --seed 1 --no_disc  --f "chi" 

