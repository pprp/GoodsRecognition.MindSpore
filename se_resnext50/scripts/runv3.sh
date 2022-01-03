#!/bin/bash
module load anaconda/2021.05
module load cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate MS


# sgd + multistep + small lr + whole dataset + 16 epochs
# python train.py > sgd_multistep_0.0001_wholedataset_16epochs.log 

# baseline + wider(from 2048->4096 linear layer) se_resnext
# python train.py --model_name se_resnext50_wider > se_resnext50_wider.log

# python train.py --model_name se_resnext50 \
#                 --data_path "/home/pdluser/datasets/all/train" \
#                 --eval_data_path "/home/pdluser/datasets/all/test" --optimizer "sgd" --lr 0.01 --num_classes 1864 --image_size 112 112 > exp1.log

# python train.py --model_name inceptionv4 --data_path "/home/pdluser/datasets/all/train" --eval_data_path "/home/pdluser/datasets/all/test" --optimizer "adam" --lr 0.0001 --num_classes 1864 --image_size 224 224 > inceptionv4.log 

# python train.py --model_name se_resnext50 --data_path "/home/pdluser/datasets/all/train" --eval_data_path "/home/pdluser/datasets/all/test" --optimizer "sgd" --lr 0.01 --num_classes 1864 --lr_gamma 0.5 --image_size 112 112 > exp3.log 

# python train.py --model_name se_resnext50 --data_path "/home/pdluser/datasets/all/train"--eval_data_path "/home/pdluser/datasets/all/test" --optimizer "adam" --lr 0.0001 --lr_gamma 0.5  --num_classes 1864 --image_size 112 112 > exp4.log 

# python train.py --model_name se_resnext50 --auto_augment True --cutout True --lr 0.05 --batch_size 128 --num_classes 1864 --data_path "/home/pdluser/datasets/all/train" --eval_data_path "/home/pdluser/datasets/all/test"  > autoaugment_test.log

# autoaugmentation test with 20% dataset 
# python train.py --model_name se_resnext50 --auto_augment True --cutout True --lr 0.05 --batch_size 128 --num_classes 1864 --data_path "/HOME/scz0088/run/all_v2/train" --eval_data_path "/HOME/scz0088/run/all_v2/test"  > autoaugment_test.log

# whole dataset training select se_resnext50, se_resnext50_wider, resnet50_bam
#python train.py --model_name se_resnext50 --max_epoch 75 --cutout True  --lr 0.05 --batch_size 128 > se_resnext_whole_dataset.log 
#python train.py --model_name se_resnext50_wider  --max_epoch 75 --cutout True  --lr 0.05 --batch_size 128 > se_resnext50_wider_whole_dataset.log 
#python train.py --model_name resnet50_bam --max_epoch 75 --cutout True --lr 0.05 --batch_size 128 > resnet50_bam_whole_dataset.log 
#python train.py --model_name inception_resnet_v2 --max_epoch 75 --cutout True --lr 0.05 --batch_size 128 > inception_resnet_v2_whole_dataset.log 

# python train.py --model_name resnet50_bam --max_epoch 75 --cutout True --lr 0.05 --batch_size 128 > resnet50_bam_with90ratation_whole_dataset.log 

# python train.py --model_name resnet50_bam_wider --cutout True > resnet50_bam_wider_whole_dataset.log

python train_arcface.py --model_name resnet50 --loss_type af > resnet50_arcface_whole_dataset.log 