import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

from sklearn.model_selection import KFold
import os, glob, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-ds", "--dataset_dir", help="where to get data", default="noonan+normal", type=str)
    # parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-k", "--kfold", help="returns the number of splitting iterations in the cross-validator.", 
                        default=10, type=int)
    parser.add_argument("-n", "--names_considered", help="names for different types considered, separated by commas", 
                        default="normal,noonan", type=str)
    parser.add_argument("-g", "--gpu_id", help="gpu id to use", default="", type=str)
    parser.add_argument("-s", "--use_shuffled_kfold", help="whether to use shuffled kfold.", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, inference=True)
    
    names_considered = args.names_considered.strip().split(',')
    threshold_array = np.arange(0, 1, 0.1)
    fp_tp = {}
    accuracy = {}
    for name in names_considered:
        fp_tp[name] = [[], []] # fpr_list, tpr_list
        accuracy[name] = []

    for threshold in threshold_array:
        learner.threshold = threshold + 1.0
        
        if conf.device.type == 'cpu': # conf.device.type = 'cpu' for CRC01/02 
            learner.load_state(conf, 'mobilefacenet.pth', True, True)
            # learner.load_state(conf, 'cpu_final.pth', True, True)
        else:
            learner.load_state(conf, 'final.pth', True, True)
        learner.model.eval()
        print('learner loaded for threshold', threshold)
        
        # mkdir for folder containing verification results
        th = 'th_' + str(threshold).replace('.', '_')
        verify_type = 'verify'
        if args.tta:
            verify_type += '_tta'
        if args.use_shuffled_kfold:
            verify_type += '_shuffled'
        verify_dir = conf.data_path/'facebank'/args.dataset_dir/verify_type/th
        if not verify_dir.is_dir():
            verify_dir.mkdir(parents=True)

        conf.facebank_path = conf.facebank_path/args.dataset_dir/'train'
        test_dir = conf.data_path/'facebank'/args.dataset_dir/'test'

        # collect raw data
        data_dict = {}
        for name in names_considered:
            data_dict[name] = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'raw') + '/' + name + '*'))

        # init kfold
        if args.use_shuffled_kfold:
            kf = KFold(n_splits=args.kfold, shuffle=True, random_state=6)
        else:
            kf = KFold(n_splits=args.kfold, shuffle=False, random_state=None)
        
        # count for roc-auc
        counts = {}
        for name in names_considered:
            counts[name] = [0, 0, 0] # #false, #true, #false_positive

        # prepare folders
        for name in names_considered:
            os.makedirs(str(conf.data_path/'facebank'/args.dataset_dir/'train') + '/' + name, exist_ok=True)
            os.makedirs(str(conf.data_path/'facebank'/args.dataset_dir/'test') + '/' + name, exist_ok=True)
            
        for fold_idx, (train_index, test_index) in enumerate(kf.split(train_set[names_considered[0]])):
            train_set = {}
            test_set = {}
            for name in names_considered:
                train_set[name], test_set[name] = data_dict[name][train_index], data_dict[name][test_index]

            # remove previous data 
            prev = glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'train') + '/*/*')
            for p in prev:
                os.remove(p)
            prev = glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'test') + '/*/*')
            for p in prev:
                os.remove(p)
            # save trains to conf.facebank_path/args.dataset_dir/'train' and 
            # tests to conf.data_path/'facebank'/args.dataset_dir/'test'
            for name in names_considered:
                for i in range(train_index.size):
                    shutil.copy(test_set[name][i], test_set[name][i].replace('raw', 'train/' + name))
                    shutil.copy(test_set[name][i], test_set[name][i].replace('raw', 'test/' + name))
            
            print(fold_idx)
            print('datasets ready')

            # prepare_facebank
            targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
            print('facebank updated')

            # folder for 1 fold
            # verify_fold = str(fold_idx)
            verify_fold_dir = verify_dir/str(fold_idx)
            if not verify_fold_dir.is_dir():
                verify_fold_dir.mkdir(parents=True)
            
            for path in test_dir.iterdir():
                # if path.is_file():
                #     continue
                # else:
                for fil in path.iterdir():
                    # if not fil.is_file():
                    #     continue
                    # else:
                    print(fil)
                    orig_name = ''.join([i for i in fil.name.strip().split('.')[0] if not i.isdigit()])
                    if orig_name not in names_considered:
                        print("Un-considered name:", fil.name)
                        continue
                    frame = cv2.imread(str(fil))
                    image = Image.fromarray(frame)
                    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                    bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1,-1,1,1] # personal choice    
                    results, score = learner.infer(conf, faces, targets, args.tta)
                    for idx,bbox in enumerate(bboxes):
                        pred_name = names[results[idx] + 1]
                        frame = draw_box_name(bbox, pred_name + '_{:.2f}'.format(score[idx]), frame)
                        if pred_name in fil.name:
                            counts[orig_name][1] += 1
                        else:
                            counts[orig_name][0] += 1
                            if pred_name in names_considered:
                                counts[pred_name][2] += 1
                    # new_name = '_'.join(str(fil).split('/')[-2:])
                    # print(verify_dir/fil.name)
                    cv2.imwrite(str(verify_fold_dir/fil.name), frame)

        print(counts)
        for name in names_considered:
            fp_tp[name][0].append(counts[name][2] / (counts[name][1] + counts[name][2]))
            fp_tp[name][1].append(counts[name][1] / (counts[name][1] + counts[name][2]))
            accuracy[name].append(counts[name][1] / (counts[name][0] + counts[name][1]))
        
    # plots
    #(area = {0:0.2f})'.format(roc_auc[name]),
    colors = list(mcolors.TABLEAU_COLORS)
    plt.figure()
    for i in range(len(names_considered)):
        name = names_considered[i]
        if i%2 != 1:
            plt.plot(fp_tp[name][0], fp_tp[name][1], label=name+' ROC curve', 
                     color=colors[i], linewidth=4)
        else:
            plt.plot(fp_tp[name][0], fp_tp[name][1], label=name+' ROC curve', 
                     color=colors[i], linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(str(conf.data_path/'facebank'/args.dataset_dir/verify_type) + '/fp_tp.png')
    plt.close()

    plt.figure()
    for i in range(len(names_considered)):
        name = names_considered[i]
        if i%2 != 1:
            plt.plot(threshold_array + 1.0, accuracy[name], label=name+' accuracy curve',
                     color=colors[i], linewidth=4)
        else:
            plt.plot(threshold_array + 1.0, accuracy[name], label=name+' accuracy curve',
                     color=colors[i], linestyle=':', linewidth=4)
    
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.savefig(str(conf.data_path/'facebank'/args.dataset_dir/verify_type) + '/accuracy.png')
    plt.close()