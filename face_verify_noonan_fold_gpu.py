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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-ds", "--dataset_dir", help="where to get data", default="noonan+normal", type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-k", "--kfold", help="returns the number of splitting iterations in the cross-validator.", 
                        default=10, type=int)
    parser.add_argument("-g", "--gpu_id", help="gpu id to use", default="", type=str)
    parser.add_argument("-s", "--use_shuffled_kfold", help="whether to use shuffled kfold.", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    args = parser.parse_args()

    conf = get_config(False, args)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, inference=True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu': # conf.device.type = 'cpu' for CRC01/02 
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
        # learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
        # learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    # mkdir for folder containing verification results
    verify_type = 'verify'
    if args.tta:
        verify_type += '_tta'
    if args.use_shuffled_kfold:
        verify_type += '_shuffled'
    verify_dir = conf.data_path/'facebank'/args.dataset_dir/verify_type
    if not verify_dir.is_dir():
        verify_dir.mkdir(parents=True)

    conf.facebank_path = conf.facebank_path/args.dataset_dir/'train'
    test_dir = conf.data_path/'facebank'/args.dataset_dir/'test'

    normals = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'raw') + '/normal*'))
    noonans = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'raw') + '/noonan*'))
    print('normals:', normals.size)
    print('noonans:', noonans.size)

    if args.use_shuffled_kfold:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=6)
    else:
        kf = KFold(n_splits=args.kfold, shuffle=False, random_state=None)
    
    # count for roc-auc
    counts = {'normal': [0, 0], 'noonan': [0, 0]} # #false, #true

    # prepare folders
    os.mkdirs(str(conf.data_path/'facebank'/args.dataset_dir/'train'/'normal'), exist_ok=True)
    os.mkdirs(str(conf.data_path/'facebank'/args.dataset_dir/'test'/'test'), exist_ok=True)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(normals)):
        normals_train, normals_test = normals[train_index], normals[test_index]
        noonans_train, noonans_test = noonans[train_index], noonans[test_index]
        # only remove the files
        prev = glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'train') + '/*/*')
        for p in prev:
            os.remove(p)
        # if os.path.exists(str(conf.data_path/'facebank'/args.dataset_dir/'train')):
        #     shutil.rmtree(str(conf.data_path/'facebank'/args.dataset_dir/'train'))
        # train_normal_dir.mkdir(parents=True)
        # train_noonan_dir.mkdir(parents=True)

        # save trains to conf.facebank_path/args.dataset_dir/'train'
        for i in range(len(normals_train)):
            shutil.copy(normals_train[i], normals_train[i].replace('raw', 'train/normal'))
            shutil.copy(noonans_train[i], noonans_train[i].replace('raw', 'train/noonan'))

        # only remove the files        
        prev = glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'test') + '/*/*')
        for p in prev:
            os.remove(p)
        # if os.path.exists(str(conf.data_path/'facebank'/args.dataset_dir/'test')):
        #     shutil.rmtree(str(conf.data_path/'facebank'/args.dataset_dir/'test'))
        # test_normal_dir.mkdir(parents=True)
        # test_noonan_dir.mkdir(parents=True)

        # save tests to conf.data_path/'facebank'/args.dataset_dir/'test'
        for i in range(len(normals_test)):
            shutil.copy(normals_test[i], normals_test[i].replace('raw', 'test/normal'))
            shutil.copy(noonans_test[i], noonans_test[i].replace('raw', 'test/noonan'))
        print(fold_idx)
        print(test_index.size)
        print('datasets ready')

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
                        counts[pred_name][1] += 1
                    else:
                        orig_name = ''.join([i for i in fil.name.split('.')[0] if not i.isdigit()])
                        counts[orig_name][0] += 1
                # new_name = '_'.join(str(fil).split('/')[-2:])
                # print(verify_dir/fil.name)
                cv2.imwrite(str(verify_fold_dir/fil.name), frame)

    print(counts)