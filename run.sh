#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N divi_lag_gpu         # Specify job name

module load python pytorch

DataDir=divided
LagData=LAG_y_fine
Model=inn05_112

for Op in "train" "test" "train,test"
do
    python fold_cur.py -ds ${DataDir} -g 0 \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}

    python fold_cur.py -ds ${DataDir} -g 0 -s \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s

    python fold_cur.py -ds ${DataDir} -g 0 -s -rs 888\
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s888

    python fold_cur.py -ds ${DataDir} -g 0 -tta \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_tta
done

for Op in "test" "train,test"
do
    python fold_cur.py -ds ${DataDir} -g 0 \
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_${Op}

    python fold_cur.py -ds ${DataDir} -g 0 -s \
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_${Op}_s

    python fold_cur.py -ds ${DataDir} -g 0 -s -rs 888\
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_${Op}_s888

    python fold_cur.py -ds ${DataDir} -g 0 -tta \
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_${Op}_tta
done


# DataDir=divided
# LagData=LAG_y_fine

# for Model in srm112df srm112df_no0
# do
#     python fold_cur.py -ds ${DataDir} -g 0 \
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_test

#     python fold_cur.py -ds ${DataDir} -g 0 -s \
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_test_s

#     python fold_cur.py -ds ${DataDir} -g 0 -s -rs 888\
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_test_s888

#     python fold_cur.py -ds ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_test_tta
# done

# for Model in srm112df srm112df_no0
# do
#     python fold_cur.py -ds ${DataDir} -g 0 \
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "train,test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_train,test

#     python fold_cur.py -ds ${DataDir} -g 0 -s \
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "train,test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_train,test_s

#     python fold_cur.py -ds ${DataDir} -g 0 -s -rs 888\
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "train,test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_train,test_s888

#     python fold_cur.py -ds ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta "test" \
#     -as ${Model} -ts "train,test" \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_${Model}_train,test_tta
# done

# DataDir=divided
# python fold_cur.py -d ${DataDir} -g 0 \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}

# python fold_cur.py -d ${DataDir} -g 0 -s \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_s

# python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_s888

# python fold_cur.py -d ${DataDir} -g 0 -tta \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_tta

# DataDir=divided
# LagData=LAG_y_fine
# Model=srm112df_no0
# #srm112d_nowrong

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}

#     python fold_cur.py -d ${DataDir} -g 0 -s \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s

#     python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s888

#     python fold_cur.py -d ${DataDir} -g 0 -tta \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_tta
# done

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}

#     python fold_cur.py -d ${DataDir} -g 0 -s \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_s

#     python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_s888

#     python fold_cur.py -d ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_tta
# done

# Model=srm112df
# #_no0wrong

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}

#     python fold_cur.py -d ${DataDir} -g 0 -s \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s

#     python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s888

#     python fold_cur.py -d ${DataDir} -g 0 -tta \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_tta
# done

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}

#     python fold_cur.py -d ${DataDir} -g 0 -s \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_s

#     python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_s888

#     python fold_cur.py -d ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_tta
# done

# TransDepth=1
# Model=smile_refine_mtcnn_112_divi
# echo ${TransDepth}
# for Epoch in 50 100
# do
#     echo ${Epoch}
#     for DataDir in distinct divided
#     do
#         echo ${DataDir}
#         python fold_cur_trans_e.py -d ${DataDir} -e ${Epoch} -g 0 -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_e${Epoch}
#         python fold_cur_trans_e.py -d ${DataDir} -e ${Epoch} -g 0 -s -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_e${Epoch}_s
#         python fold_cur_trans_e.py -d ${DataDir} -e ${Epoch} -g 0 -tta -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_e${Epoch}_tta
#     done
# done

# TransDepth=1
# Model=smile_refine_mtcnn_112_divi
# echo ${TransDepth}
# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     for Op in "train" "test" "train,test"
#     do
#         echo ${Op}
#         python fold_cur_trans.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}
#         python fold_cur_trans.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_s
#         python fold_cur_trans.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_tta
#     done
# done

# DataDir=divided
# AugDir=LAG_y_fine
# echo ${DataDir}
# echo ${AugDir}
# python roc_acc_fold_cur.py -d ${DataDir} -sd data/facebank/${DataDir}/plt_recs \
# -a ${AugDir} > data/facebank/${DataDir}/plt_recs/${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} -s -sd data/facebank/${DataDir}/plt_recs \
# -a ${AugDir} > data/facebank/${DataDir}/plt_recs/${DataDir}_s
# python roc_acc_fold_cur.py -d ${DataDir} -tta -sd data/facebank/${DataDir}/plt_recs \
# -a ${AugDir} > data/facebank/${DataDir}/plt_recs/${DataDir}_tta

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
