#!/bin/bash
# module load anaconda/2021.05
# module load cuda/11.1
# module load cudnn/8.2.1_cuda11.x
# source activate MS

python eval_errorcase.py --model_name resnet50_bam --data_path "/home/pdluser/datasets/all/train" --eval_data_path "/home/pdluser/datasets/all/test" --checkpoint_file_path "./trained_weights/0_model_name_resnet50_bam-75_2672.ckpt"