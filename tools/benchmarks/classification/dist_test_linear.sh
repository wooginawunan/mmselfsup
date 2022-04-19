#!/usr/bin/env bash

set -e
set -x

CFG=$1  # use cfgs under "configs/benchmarks/classification/imagenet/*.py"
CHECKPOINT=$2  # pretrained model
SAVEFOLD=$3
PY_ARGS=${@:4}
GPUS=${GPUS:-8}  # When changing GPUS, please also change samples_per_gpu in the config file accordingly to ensure the total batch size is 256.
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$SAVEFOLD"

# TORCH_DISTRIBUTED_DEBUG=DETAIL

echo $WORK_DIR
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py $CFG $CHECKPOINT \
    --work-dir "$WORK_DIR" \
    --launcher="pytorch" \
    --breast \
    ${PY_ARGS}

    
# /gpfs/data/geraslab/Nan/mmselfsup/work_dirs/benchmarks/classification/nyubreast/us/20220411_us_latest.pth/unfreeze_batch16/epoch_100.pth
# epoch_100_on_full_val

# balanced_imagenet
# epoch_100_on_full_val



