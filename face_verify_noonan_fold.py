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
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-k", "--kfold", help="Returns the number of splitting iterations in the cross-validator.", 
                        default=10, type=int)
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, inference=True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu': # conf.device.type = 'cpu' for CRC01/02 
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
        # learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    conf.facebank_path = conf.facebank_path/args.dataset_dir/'train'
    test_dir = conf.data_path/'facebank'/args.dataset_dir/'test'

    normals = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'raw') + '/normal*'))
    noonans = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'raw') + '/noonan*'))

    kf = KFold(n_splits=args.kfold)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(normals)):
        normals_train, normals_test = normals[train_index], normals[test_index]
        noonans_train, noonans_test = noonans[train_index], noonans[test_index]
        # save trains to conf.facebank_path/args.dataset_dir/'train'
        prev = glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'train') + '/*/*')
        for p in prev:
            os.remove(p)
        for i in range(len(normals_train)):
            shutil.copy(normals_train[i], normals_train[i].replace('raw', 'train/normal'))
            shutil.copy(noonans_train[i], noonans_train[i].replace('raw', 'train/noonan'))
        # save tests to conf.data_path/'facebank'/args.dataset_dir/'test'
        prev = glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/'test') + '/*/*')
        for p in prev:
            os.remove(p)
        for i in range(len(normals_test)):
            print('{} to {}'.format(normals_test[i], normals_test[i].replace('raw', 'test/normal')))
            print('{} to {}'.format(noonans_test[i], noonans_test[i].replace('raw', 'test/normal')))
            shutil.copy(normals_test[i], normals_test[i].replace('raw', 'test/normal'))
            shutil.copy(noonans_test[i], noonans_test[i].replace('raw', 'test/noonan'))
        print(fold_idx)
        print('datasets ready')

        # targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        # print('facebank updated')

        if args.tta:
            verify_type = 'verify_tta_' + str(fold_idx)
        else:
            verify_type = 'verify_' + str(fold_idx)
        verify_dir = conf.data_path/'facebank'/args.dataset_dir/'fold'/verify_type
        if not verify_dir.is_dir():
            verify_dir.mkdir(parents=True)
        
        for path in test_dir.iterdir():
            if path.is_file():
                continue
            else:
                for fil in path.iterdir():
                    if not fil.is_file():
                        continue
                    else:
                        print(fil)
                        # frame = cv2.imread(str(fil))
                        # image = Image.fromarray(frame)
                        # bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                        # bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                        # bboxes = bboxes.astype(int)
                        # bboxes = bboxes + [-1,-1,1,1] # personal choice    
                        # results, score = learner.infer(conf, faces, targets, args.tta)
                        # for idx,bbox in enumerate(bboxes):
                        #     frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                        
                        # # new_name = '_'.join(str(fil).split('/')[-2:])
                        # # print(verify_dir/fil.name)
                        # cv2.imwrite(str(verify_dir/fil.name), frame)


