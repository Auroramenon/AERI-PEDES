#!/bin/bash
DATASET_NAME="AERI-PEDES"    # AERI-PEDES  AGDataAttr

CUDA_VISIBLE_DEVICES=5 \
python finetune.py \
--name cda-slot-cross-attn-k8 \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'cda' \
--avm_mode 'slot_cross' \
--avm_num_slots 8 \
--avm_loss_weight 1.0 \
--avm_lr 1e-4 \
--lr 5e-6 \
--lr2 5e-5 \
--num_epoch 60 \
--root_dir /data1/Datasets/ReID/ \
--finetune 'pretrain/HAMbest0.pth'
