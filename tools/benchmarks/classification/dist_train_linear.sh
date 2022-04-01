#!/usr/bin/env bash

set -e
set -x

CFG=$1  # use cfgs under "configs/benchmarks/classification/imagenet/*.py"
PRETRAIN=$2  # pretrained model
# PY_ARGS=${@:3}
GPUS=$3  # When changing GPUS, please also change samples_per_gpu in the config file accordingly to ensure the total batch size is 256.
PORT=${PORT:-29500}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG \
    #--cfg-options model.ffdm_backbone.init_cfg.type=Pretrained \
    model.ffdm_backbone.init_cfg.checkpoint=$PRETRAIN \
    --work_dir $WORK_DIR --seed 0 --launcher="pytorch" --breast


# bash tools/benchmarks/classification/dist_train_liner.sh \
# configs/benchmarks/classification/nyubreast/ffdm.py \
# /gpfs/data/geraslab/Nan/saves/selfsup/swav_breast/data_20220111_full/swav_resnet18_avgpool_coslr-100e_largebatch_skynet-gpu32/latest.pth
# 1 

