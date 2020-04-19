#!/bin/bash

#$ -M jliu16@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -pe smp 4	 # Specify parallel environment and legal core size
#$ -q long		 # Specify queue
#$ -N verify_test	         # Specify job name

module load python pytorch

DataDir=divided
echo ${DataDir}
python roc_acc_fold_cur.py -d ${DataDir} -sd data/facebank/${DataDir}/plt_recs \
> data/facebank/${DataDir}/plt_recs/${DataDir}
python roc_acc_fold_cur.py -d ${DataDir} -s -sd data/facebank/${DataDir}/plt_recs \
> data/facebank/${DataDir}/plt_recs/${DataDir}_s
python roc_acc_fold_cur.py -d ${DataDir} -tta -sd data/facebank/${DataDir}/plt_recs \
> data/facebank/${DataDir}/plt_recs/${DataDir}_tta
# python roc_acc_fold_cur.py -d ${DataDir} > data/facebank/plt_recs/${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} -s > data/facebank/plt_recs/${DataDir}_s
# python roc_acc_fold_cur.py -d ${DataDir} -tta > data/facebank/plt_recs/${DataDir}_tta
# for Model in mix_aug_500_pool5_full_refine mix_aug_500_full_refine mix_aug_100_pool5_full_refine
# do
#     echo ${Model}
#     python roc_acc_fold_cur.py -d ${DataDir} -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#     python roc_acc_fold_cur.py -d ${DataDir} -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
# done

# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} > data/facebank/plt_recs/${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -s > data/facebank/plt_recs/${DataDir}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -tta > data/facebank/plt_recs/${DataDir}_tta
#     for Model in mix_aug_500_pool5_full_refine mix_aug_500_full_refine mix_aug_100_pool5_full_refine
#     do
#         echo ${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#         python roc_acc_fold_cur.py -d ${DataDir} -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
#     done
# done

# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} > data/facebank/plt_recs/${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -s > data/facebank/plt_recs/${DataDir}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -tta > data/facebank/plt_recs/${DataDir}_tta
#     for Model in fr_mix_aug_pix2pix_transfer_b6_500_pool5_full fr_mix_aug_pix2pix_transfer_b6_500_full fr_mix_aug_pix2pix_transfer_b6_100_pool5_full fr_mix_aug_pix2pix_transfer_b6_100_full
#     do
#         echo ${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#         python roc_acc_fold_cur.py -d ${DataDir} -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
#     done
# done

# python evaluate_model.py

# for Model in fr_mix_aug_pix2pix_transfer_b6_500_pool5_full fr_mix_aug_pix2pix_transfer_b6_500_full
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

# for Model in fr_mix_aug_pix2pix_transfer_b6_100_pool5_full fr_mix_aug_pix2pix_transfer_b6_100_full
# do
#     echo "${Model}"
#     python roc_acc_fold.py -d divided -g 0 -a fake_${Model} > data/facebank/roc_divi_${Model}
#     tail -n 2 data/facebank/roc_divi_${Model}
#     echo "${Model}_s"
#     python roc_acc_fold.py -d divided -g 0 -s -a fake_${Model} > data/facebank/roc_divi_${Model}_s
#     tail -n 2 data/facebank/roc_divi_${Model}_s
#     echo "${Model}_tta"
#     python roc_acc_fold.py -d divided -g 0 -tta -a fake_${Model} > data/facebank/roc_divi_${Model}_tta
#     tail -n 2 data/facebank/roc_divi_${Model}_tta
# done

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
