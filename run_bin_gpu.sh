#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N bin_test_gpu         # Specify job name

module load pytorch

python bin_roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s > data/facebank/bin_s_0
# python bin_roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES > data/facebank/bin_0