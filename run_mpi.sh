#!/bin/bash

#$ -M jliu16@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -pe mpi-24 24	 # Specify parallel environment and legal core size
#$ -q long		 # Specify queue
#$ -N verify_test_mpi	         # Specify job name

module load python

python face_verify_noonan_fold.py > data/facebank/default_2_mpi
python face_verify_noonan_fold.py -tta > data/facebank/tta_1_mpi
python face_verify_noonan_fold.py -s > data/facebank/shuffle_1_mpi
python face_verify_noonan_fold.py -s -tta > data/facebank/shuffle_tta_1_mpi
