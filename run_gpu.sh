#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N verify_test_gpu         # Specify job name

module load pytorch

python face_verify_noonan_fold_gpu.py -g $CUDA_VISIBLE_DEVICES > data/facebank/default_2_gpu
python face_verify_noonan_fold_gpu.py -tta -g $CUDA_VISIBLE_DEVICES > data/facebank/tta_1_gpu
python face_verify_noonan_fold_gpu.py -s -g $CUDA_VISIBLE_DEVICES > data/facebank/shuffle_1_gpu
python face_verify_noonan_fold_gpu.py -s -tta -g $CUDA_VISIBLE_DEVICES > data/facebank/shuffle_tta_1_gpu