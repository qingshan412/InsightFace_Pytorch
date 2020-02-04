#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N verify_test_gpu         # Specify job name

module load python

--gpu_ids $CUDA_VISIBLE_DEVICES

python face_verify_noonan_fold_gpu.py > data/facebank/default_2
python face_verify_noonan_fold_gpu.py -tta > data/facebank/tta_1
python face_verify_noonan_fold_gpu.py -s > data/facebank/shuffle_1
python face_verify_noonan_fold_gpu.py -s -tta > data/facebank/shuffle_tta_1
