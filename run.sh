#!/bin/bash

#$ -M jliu16@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -pe smp 4	 # Specify parallel environment and legal core size
#$ -q long		 # Specify queue
#$ -N verify_test	         # Specify job name

module load python

module load pytorch

for Model in fr_mix_aug_pix2pix_transfer_b6_100_pool5_full fr_mix_aug_pix2pix_transfer_b6_100_full
do
    echo "${Model}"
    python roc_acc_fold.py -d divided -g 0 -a fake_${Model} > data/facebank/roc_divi_${Model}
    tail -n 2 data/facebank/roc_divi_${Model}
    echo "${Model}_s"
    python roc_acc_fold.py -d divided -g 0 -s -a fake_${Model} > data/facebank/roc_divi_${Model}_s
    tail -n 2 data/facebank/roc_divi_${Model}_s
    echo "${Model}_tta"
    python roc_acc_fold.py -d divided -g 0 -tta -a fake_${Model} > data/facebank/roc_divi_${Model}_tta
    tail -n 2 data/facebank/roc_divi_${Model}_tta
done

# Model=fr_mix_aug_pix2pix_transfer_b6_100_pool5_full
# echo "${Model}"
# python roc_acc_fold.py -d distinct -g 0 -a fake_${Model} > data/facebank/roc_dist_${Model}
# tail -n 2 data/facebank/roc_dist_${Model}
# echo "${Model}_s"
# python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Model} > data/facebank/roc_dist_${Model}_s
# tail -n 2 data/facebank/roc_dist_${Model}_s
# echo "${Model}_tta"
# python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Model} > data/facebank/roc_dist_${Model}_tta
# tail -n 2 data/facebank/roc_dist_${Model}_tta


# for Model in fr_mix_aug_pix2pix_transfer_b6_2000_DG fr_mix_pix2pix_transfer_b6_2000_DG
# do
#     echo "${Model}"
#     python roc_acc_fold.py -d distinct -g 0 -a fake_${Model} > data/facebank/roc_dist_${Model}
#     tail -n 2 data/facebank/roc_dist_${Model}
#     echo "${Model}_s"
#     python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Model} > data/facebank/roc_dist_${Model}_s
#     tail -n 2 data/facebank/roc_dist_${Model}_s
#     echo "${Model}_tta"
#     python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Model} > data/facebank/roc_dist_${Model}_tta
#     tail -n 2 data/facebank/roc_dist_${Model}_tta
# done


# python face_verify_noonan_fold.py > data/facebank/default_2
# python face_verify_noonan_fold.py -tta > data/facebank/tta_1
# python face_verify_noonan_fold.py -s > data/facebank/shuffle_1
# python face_verify_noonan_fold.py -s -tta > data/facebank/shuffle_tta_1
