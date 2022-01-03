#!/bin/bash
# module load anaconda/2021.05
# module load cuda/11.1
# module load cudnn/8.2.1_cuda11.x
# source activate MS




# fixing train-test resolution
# training 75 epoch
# python train.py --model-name se_resnext50 --image_size 112 112 --max_epoch 25 --lr 0.4 --T_max 25 > se_resnext50_withratio_112_lr0.4.out
# python train.py --model-name se_resnext50 --image_size 112 112 --max_epoch 25 --lr 0.2 --T_max 25 > se_resnext50_withratio_112_lr0.2.out
# python train.py --model-name se_resnext50 --image_size 112 112 --max_epoch 25 --lr 0.1 --T_max 25 > se_resnext50_withratio_112_lr0.2.out


# fintune 5 epoch
# python finetune.py --model-name se_resnext50 --image_size 256 256 --max_epoch 5 --lr 0.04 --T_max 5 --checkpoint_file_path ??? > se_resnext50_withratio_finetune256.out

# image size with ratio resize
# python train.py --model-name se_resnet50 --image_size 224 112 > resnet50_bam_112_withratio.out
# python train.py --model-name se_resnet50 --image_size 224 88 > resnet50_bam_88_withratio.out
# python train.py --model-name se_resnet50 --image_size 224 72 > resnet50_bam_72_withratio.out

# image size with normal resize
# python train.py --model-name se_resnet50 --image_size 224 112 > resnet50_bam_112.out
# python train.py --model-name se_resnet50 --image_size 224 88 > resnet50_bam_88.out
# python train.py --model-name se_resnet50 --image_size 224 72 > resnet50_bam_72.out

# cutout experiment
# python train.py --model-name se_resnext50 --cutout True --cutout-length 56 > se_resnext50_cutout56.out
# python train.py --model-name se_resnext50 --cutout True --cutout-length 112 > se_resnext50_cutout112.out
 
# drop out baseline
# python train.py --model-name se_resnext50 > se_resnext50_baseline.out
# 重新

# base model test
# python train.py --model-name resnet50_bam > resnet50_bam.out
# 14G
# python train.py --model-name resnet50_cbam > resnet50_cbam.out 
# 18G
# python train.py --model-name se_resnet50 > se_resnet50.out
# 14G
# python train.py --model-name resnet101 > resnet101.out
# 16G
# python train.py --model-name resnet50 > resnet50.out
# 12G

# OOM
# python train.py --model-name efficientnetb4 > efficientnetb4.out
# python train.py --model-name wideresnet_d4w10 > wideresnet_d4w10.out
# python train.py --model-name resnext101 > resnext101.out
