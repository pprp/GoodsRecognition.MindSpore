#!/bin/bash

export DEVICE_ID=$1
DATA_DIR=$2
EVAL_DATA_DIR=$3
MODEL_NAME=$4


cd ..

python train.py  \
    --run_distribute=0 \
    --device_id=$DEVICE_ID \
    --model_name=$MODEL_NAME \
    --data_path=$DATA_DIR \
    --eval_data_path=$EVAL_DATA_DIR \
    --output_path './output' > $MODEL_NAME.txt 2>&1 & 

