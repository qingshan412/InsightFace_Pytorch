#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N verify_test_gpu         # Specify job name

module load pytorch

for Epoch in 20 latest
do
    for Type in 'raw' 'resize'# 'raw_resize'
    do
        for Model in 'fr_adult_pix2pix_transfer_b6_25_1layer_fe' 'fr_adult_pix2pix_transfer_b6_25_2layer' 'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG'
        do
            python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -a fake_${Type}_${Epoch}_${Model} \
            > data/facebank/roc_${Type}_${Epoch}_${Model}
            python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -a fake_${Type}_${Epoch}_${Model} \
            > data/facebank/roc_${Type}_${Epoch}_${Model}_s
            python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta -a fake_${Type}_${Epoch}_${Model} \
            > data/facebank/roc_${Type}_${Epoch}_${Model}_tta
        done
    done
done

# for Epoch in 10 15
# do
#     for Type in 'raw' 'resize' 'raw_resize'
#     do
#         for Model in '1layer_fe' '2layer' '1layer_pool5_DG'
#         do
#             echo ${Type}_${Epoch}_${Model}
#             tail -n 3 roc_${Type}_${Epoch}_fr_adult_pix2pix_transfer_b6_25_${Model}
#             echo ${Type}_${Epoch}_${Model}_s
#             tail -n 3 roc_${Type}_${Epoch}_fr_adult_pix2pix_transfer_b6_25_${Model}_s
#             echo echo ${Type}_${Epoch}_${Model}_tta
#             tail -n 3 roc_${Type}_${Epoch}_fr_adult_pix2pix_transfer_b6_25_${Model}_tta
#         done
#     done
# done

# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -a "fake_raw" > data/facebank/roc_raw_112_s_a_raw
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta -a "fake_raw" > data/facebank/roc_raw_112_tta_a_raw
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta \
#     -a "fake_raw" > data/facebank/roc_raw_112_s_tta_a_raw
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -a "fake_raw" > data/facebank/roc_raw_112_a_raw

# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -a "fake_resize" > data/facebank/roc_raw_112_a_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -a "fake_resize" > data/facebank/roc_raw_112_s_a_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta -a "fake_resize" > data/facebank/roc_raw_112_tta_a_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta -a "fake_resize" > data/facebank/roc_raw_112_s_tta_a_resize

# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -a "fake_noonan_resize" > data/facebank/roc_raw_112_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s \
#     -a "fake_noonan_resize" > data/facebank/roc_raw_112_s_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta \
#     -a "fake_noonan_resize" > data/facebank/roc_raw_112_tta_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta \
#     -a "fake_noonan_resize" > data/facebank/roc_raw_112_s_tta_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES > data/facebank/roc_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s > data/facebank/roc_s_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta > data/facebank/roc_tta_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta > data/facebank/roc_s_tta_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s > data/facebank/roc_s_0
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta > data/facebank/roc_tta_0
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta > data/facebank/roc_s_tta_0
# python face_verify_noonan_fold_gpu.py -g $CUDA_VISIBLE_DEVICES > data/facebank/default_2_gpu
# python face_verify_noonan_fold_gpu.py -tta -g $CUDA_VISIBLE_DEVICES > data/facebank/tta_1_gpu
# python face_verify_noonan_fold_gpu.py -s -g $CUDA_VISIBLE_DEVICES > data/facebank/shuffle_1_gpu
# python face_verify_noonan_fold_gpu.py -s -tta -g $CUDA_VISIBLE_DEVICES > data/facebank/shuffle_tta_1_gpu
