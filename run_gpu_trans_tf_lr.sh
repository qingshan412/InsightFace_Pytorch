#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N lr_test         # Specify job name

module load pytorch

TransDepth=2
Model=smile_refine_mtcnn_112_divi
echo ${TransDepth}
for DataDir in divided distinct
do
    echo ${DataDir}
    python fold_cur_trans_tf_lr.py -d ${DataDir} -g 0 -t ${TransDepth} \
    > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_Adam_lr4
    python fold_cur_trans_tf_lr.py -d ${DataDir} -g 0 -s -t ${TransDepth} \
    > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_Adam_lr4_s
    python fold_cur_trans_tf_lr.py -d ${DataDir} -g 0 -tta -t ${TransDepth} \
    > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_Adam_lr4_tta
    for Op in "test" "train,test"
    do
        echo ${Op}
        python fold_cur_trans_tf_lr.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} -t ${TransDepth} \
        > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_Adam_lr4
        python fold_cur_trans_tf_lr.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} -t ${TransDepth} \
        > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_Adam_lr4_s
        python fold_cur_trans_tf_lr.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} -t ${TransDepth} \
        > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_Adam_lr4_tta
    done
done


# TransDepth=1
# Model=smile_refine_mtcnn_112_divi
# echo ${TransDepth}
# for DataDir in divided distinct
# do
#     echo ${DataDir}
#     Op="train,test"
#     echo ${Op}
#     python fold_cur_trans_tf.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} -t ${TransDepth} \
#     > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_Adam
#     python fold_cur_trans_tf.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} -t ${TransDepth} \
#     > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_Adam_s
#     python fold_cur_trans_tf.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} -t ${TransDepth} \
#     > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_Adam_tta
# done

