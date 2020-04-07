#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N verify_test_gpu         # Specify job name

module load pytorch

for DataDir in distinct divided
do
    for Model in mix_aug_500_pool5_full_refine mix_aug_500_full_refine mix_aug_100_pool5_full_refine
    do
        echo ${Model}
        python roc_acc_fold_cur.py -d ${DataDir} -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
        python roc_acc_fold_cur.py -d ${DataDir} -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
        python roc_acc_fold_cur.py -d ${DataDir} -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
    done
done

python evaluate_model.py

# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s > data/facebank/plt_recs/${DataDir}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta > data/facebank/plt_recs/${DataDir}_tta
#     for Model in fr_mix_aug_pix2pix_transfer_b6_500_pool5_full fr_mix_aug_pix2pix_transfer_b6_500_full fr_mix_aug_pix2pix_transfer_b6_100_pool5_full fr_mix_aug_pix2pix_transfer_b6_100_full
#     do
#         echo ${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
#     done
# done

# for Model in fr_mix_aug_pix2pix_transfer_b6_100_full fr_mix_aug_pix2pix_transfer_b6_2000_D2G fr_mix_pix2pix_transfer_b6_2000_D2G
# do
#     python roc_acc_fold.py -d distinct -g 0 -a fake_${Model} > data/facebank/roc_dist_${Model}
#     python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Model} > data/facebank/roc_dist_${Model}_s
#     python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Model} > data/facebank/roc_dist_${Model}_tta
# done

# Epoch=latest

# for Type in 'raw' 'resize' 'raw_resize'
# do
#     for Model in 'fr_adult_pix2pix_transfer_b6_25_1layer_fe' 'fr_adult_pix2pix_transfer_b6_25_2layer' 'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG'
#     do
#         python roc_acc_fold.py -d distinct -g 0 -a fake_${Type}_${Epoch}_${Model} \
#         > data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#         tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#         python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Type}_${Epoch}_${Model} \
#         > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#         tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#         python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Type}_${Epoch}_${Model} \
#         > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#         tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#     done
# done


# for Epoch in 20 latest
# do
#     for Type in 'raw' 'resize' 'raw_resize'
#     do
#         for Model in 'fr_adult_pix2pix_transfer_b6_25_1layer_fe' 'fr_adult_pix2pix_transfer_b6_25_2layer' 'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG'
#         do
#             echo "${Type}_${Epoch}_${Model}"
#             python roc_acc_fold.py -d distinct -g 0 -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#             tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#             echo "${Type}_${Epoch}_${Model}_s"
#             python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#             tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#             echo "${Type}_${Epoch}_${Model}_tta"
#             python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#             tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#         done
#     done
# done


# for Model in raw_fr_lag_aug_pix2pix_transfer_b6_100_2layer fr_lag_aug_pix2pix_transfer_b6_100_2layer fr_lag_pix2pix_transfer_b6_100_2layer fr_lag_aug_pix2pix_transfer_b6_500_2layer fr_mix_aug_pix2pix_transfer_b6_500_DG fr_mix_pix2pix_transfer_b6_500_DG
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

# python roc_acc_fold.py -g 0 -s -a fake_fr_aug_pix2pix_transfer_b6_100_2layer > data/facebank/roc_aug_100_2layer_s

# for Epoch in 20 latest
# do
#     for Type in 'raw' 'resize' # 'raw_resize'
#     do
#         for Model in 'fr_adult_pix2pix_transfer_b6_25_1layer_fe' 'fr_adult_pix2pix_transfer_b6_25_2layer' 'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG'
#         do
#             python roc_acc_fold.py -g 0 -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_${Type}_${Epoch}_${Model}
#             python roc_acc_fold.py -g 0 -s -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_${Type}_${Epoch}_${Model}_s
#             python roc_acc_fold.py -g 0 -tta -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_${Type}_${Epoch}_${Model}_tta
#         done
#     done
# done

#10 15
#'raw_resize'
# for Epoch in 20 latest
# do
#     for Type in 'raw' 'resize'
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
