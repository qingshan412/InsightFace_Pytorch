import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank, save_label_score

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
import os, glob, shutil, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-ds", "--dataset_dir", help="where to get data", default="srm112d_no0", type=str)
    parser.add_argument('-sd','--stored_data_dir',help='where to store data as np arrays',default="data/facebank/plt_recs", type=str)
    # parser.add_argument("-k", "--kfold", help="returns the number of splitting iterations in the cross-validator.", 
    #                     default=10, type=int)
    parser.add_argument("-n", "--names_considered", help="names for different types considered, separated by commas", 
                        default="normal,noonan", type=str)
    parser.add_argument("-g", "--gpu_id", help="gpu id to use", default="", type=str)
    # parser.add_argument("-s", "--use_shuffled_kfold", help="whether to use shuffled kfold.", action="store_true")
    parser.add_argument("-rs", "--random_seed", help="init random seed.", default=6, type=int)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-a", "--additional_data_dir", help="where to get the additional data", 
                        default="", type=str)
    parser.add_argument("-ta", "--additional_test_or_train", help="use additional data in only train, or test, or both", 
                        default="", type=str)
    # parser.add_argument("-as", "--stylegan_data_dir", help="where to get the additional data, "
    #                     "not only for the stylegan", default="", type=str)
    # parser.add_argument("-ts", "--stylegan_test_or_train", help="use stylegan data in only train, or test, or both", 
    #                     default="", type=str)
    args = parser.parse_args()

    random.seed(args.random_seed)

    conf = get_config(False, args)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    names_considered = args.names_considered.strip().split(',')

    if 'smile' in args.dataset_dir:
        exp_name = 'smile'
    elif 'no0' in args.dataset_dir:
        exp_name = 'sn0' #srm112d_no0
    else:
        exp_name = args.dataset_dir
    if args.additional_data_dir:
        if 'LAG' in args.additional_data_dir:
            exp_name += '_lag'
        else:
            exp_name += ('_' + args.additional_data_dir)
        exp_name += ('_' + args.additional_test_or_train)
        exp_name += ('_rs' + str(args.random_seed))
    if args.tta:
        exp_name += '_tta'
    
    # if args.stylegan_data_dir:
    #     #e.g. smile_refine_mtcnn_112_divi
    #     full_stylegan_dir = str(conf.data_path/'facebank'/'stylegan'/args.stylegan_data_dir)
    #     stylegan_folders = os.listdir(full_stylegan_dir)


    # prepare folders
    raw_dir = 'raw_112' #'mtcnn_112_aug'
    verify_type = 'reverse' #'mtcnn_112_aug' 'verify'
    if args.tta:
        verify_type += '_tta'

    train_dir = conf.facebank_path/args.dataset_dir/verify_type/'train'
    test_dir = conf.facebank_path/args.dataset_dir/verify_type/'test'
    conf.facebank_path = train_dir
    for name in names_considered:
        os.makedirs(str(train_dir) + '/' + name, exist_ok=True)
        os.makedirs(str(test_dir) + '/' + name, exist_ok=True)

    # collect raw data
    data_dict = {}
    test_dict = {}
    for name in names_considered:
        data_dict[name] = np.array(glob.glob(str(conf.data_path/'facebank'/'stylegan'/args.dataset_dir) + 
                                            '/' + name + '*'))
        test_dict[name] = np.array(glob.glob(str(conf.data_path/'facebank'/'divided'/raw_dir) + 
                                            '/' + name + '*'))

    if 'LAG' in args.additional_data_dir:
        full_additional_dir = conf.data_path/'facebank'/args.additional_data_dir/raw_dir
        data_dict['lag'] = np.array(glob.glob(str(full_additional_dir) + '/*'))

    learner = face_learner(conf, inference=True)
    
    if conf.device.type == 'cpu': # conf.device.type = 'cpu' for CRC01/02 
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
        # learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
    learner.model.eval()
    print('learner loaded.')

    # # count for roc-auc
    # counts = {}
    # for name in names_considered:
    #     counts[name] = [0, 0, 0] # #false, #true, #false_positive
    score_names = []
    scores = []

    train_set = {}
    test_set = {}
    for name in names_considered:
        train_set[name], test_set[name] = data_dict[name][:], test_dict[name][:]


    if 'lag' in data_dict.keys():
        if 'train' in args.additional_test_or_train and 'test' in args.additional_test_or_train:
            lag_size = data_dict['lag'].shape[0]
            train_idx = random.sample([i for i in range(lag_size)], lag_size//2)
            test_idx = [i for i in range(lag_size) if i not in train_idx]
            train_set['normal'] = np.concatenate((train_set['normal'], data_dict['lag'][train_idx]))
            test_set['normal'] = np.concatenate((test_set['normal'], data_dict['lag'][test_idx]))
        elif 'train' in args.additional_test_or_train:
            train_set['normal'] = np.concatenate((train_set['normal'], data_dict['lag']))
        elif 'test' in args.additional_test_or_train:
            test_set['normal'] = np.concatenate((test_set['normal'], data_dict['lag']))

    # remove previous data 
    prev = glob.glob(str(train_dir) + '/*/*')
    for p in prev:
        os.remove(p)
    prev = glob.glob(str(test_dir) + '/*/*')
    for p in prev:
        os.remove(p)
    # save trains to conf.facebank_path/args.dataset_dir/'train' and 
    # tests to conf.data_path/'facebank'/args.dataset_dir/'test'
    for name in names_considered:
        # train
        for i in range(len(train_set[name])):
            for img in os.listdir(train_set[name][i]):
                shutil.copy(train_set[name][i] + os.sep + img, 
                            os.path.join(str(train_dir), name, img))
                            # ('/'.join(train_set[name][i].strip().split('/')[:-2]) + 
                            #     '/' + verify_type + '/train/' + name + os.sep + img))
            # # addition data from stylegan
            # folder = os.path.basename(train_set[name][i])
            # if args.stylegan_data_dir and ('train' in args.stylegan_test_or_train) and (folder in stylegan_folders):
            #     for img in os.listdir(full_stylegan_dir + os.sep + folder):
            #         shutil.copy(os.path.join(full_stylegan_dir, folder, img), 
            #                     os.path.join(str(train_dir), name, img))
            #                     # ('/'.join(train_set[name][i].strip().split('/')[:-2]) + 
            #                     #     '/' + verify_type + '/train/' + name + os.sep + img))
        # test
        for i in range(len(test_set[name])):
            for img in os.listdir(test_set[name][i]):
                shutil.copy(test_set[name][i] + os.sep + img, 
                            os.path.join(str(test_dir), name, img))
                            # ('/'.join(test_set[name][i].strip().split('/')[:-2]) + 
                            #     '/' + verify_type + '/test/' + name + os.sep + img))
            # # addition data from stylegan
            # folder = os.path.basename(test_set[name][i])
            # if args.stylegan_data_dir and ('test' in args.stylegan_test_or_train) and (folder in stylegan_folders):
            #     # and 
            #     # (folder not in ['noonan7','noonan19','noonan23','normal9','normal20','normal23'])):
            #     for img in os.listdir(full_stylegan_dir + os.sep + folder):
            #         shutil.copy(os.path.join(full_stylegan_dir, folder, img), 
            #                     os.path.join(str(test_dir), name, img))


    if 'fake' in args.additional_data_dir:
        fake_dict = {'noonan':'normal', 'normal':'noonan'}
        full_additional_dir = conf.data_path/'facebank'/'noonan+normal'/args.additional_data_dir
        add_data = glob.glob(str(full_additional_dir) + os.sep + '*.png')
        print('additional:', args.additional_data_dir, len(add_data))
        for name in names_considered:
            for img_f in add_data:
                if name in img_f.strip().split(os.sep)[-1]:
                    # print('source:', img_f)
                    # print('copy to:', img_f.replace(str(full_additional_dir), 
                    #                                 str(train_dir) + os.sep + fake_dict[name]))
                    # print('copy to:', img_f.replace(args.additional_data_dir, 
                    #                                 verify_type + '/train/' + name))
                    shutil.copy(img_f, img_f.replace(str(full_additional_dir), 
                                                    str(train_dir) + os.sep + fake_dict[name]))

    print('datasets ready')

    # prepare_facebank
    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
    print('names_classes:', names)
    names_idx = {'noonan':1, 'normal':0}
    if ('noonan' in names[1]) and ('normal' in names[2]):
        noonan_idx = 0
        # names_idx = {'noonan': 0, 'normal': 1}
    elif ('noonan' in names[2]) and ('normal' in names[1]):
        noonan_idx = 1
        # names_idx = {'noonan':1, 'normal':0}
    else:
        print('something wrong with names:', names)
        exit(0)
    print('facebank updated')

    
    for path in test_dir.iterdir():
        if path.is_file():
            continue
        # print(path)
        for fil in path.iterdir():
            # print(fil)
            orig_name = ''.join([i for i in fil.name.strip().split('.')[0].split('_')[0] if not i.isdigit()])
            if 'noonan' in orig_name:
                score_names.append(names_idx['noonan'])
            else:
                score_names.append(names_idx['normal'])
            if orig_name not in names_considered:
                print("Un-considered name:", fil.name)
                continue
            frame = cv2.imread(str(fil))
            image = Image.fromarray(frame)
            faces = [image,]
            # bboxes, _ = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            # bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            # # print('bboxes shape:', bboxes.shape)
            # bboxes = bboxes.astype(int)
            # bboxes = bboxes + [-1,-1,1,1] # personal choice
            score = learner.binfer(conf, faces, targets, args.tta)
            scores.append(score)


    score_names = np.array(score_names)
    # print(score_names.shape)
    scores_np = np.squeeze(np.array(scores))
    # print(scores_np.shape)
    relative_scores = 1 - (scores_np[:, noonan_idx] / (scores_np[:, 0] + scores_np[:, 1]))
    # print('score_names:')
    # print(score_names)
    # print('scores_np:')
    # print(relative_scores)

    # if score_names.shape[0] == 58:
    #     ext = 'dist'
    # elif score_names.shape[0] == 104:
    #     ext = 'divi'
    # elif score_names.shape[0] == 154:
    #     ext = 'styl'
    # elif score_names.shape[0] == 58 + 154:
    #     ext = 'dist+styl'
    # elif score_names.shape[0] == 104 + 154:
    #     ext = 'divi+styl'
    # else:
    #     print('label dimension wrong:',score_names.shape[0])
    #     exit(0)
    name_path = os.path.join(args.stored_data_dir, 'names_no_trans.npy')
    save_label_score(name_path, exp_name)
    label_path = os.path.join(args.stored_data_dir, 'labels_no_trans.npy')
    save_label_score(label_path, score_names)
    score_path = os.path.join(args.stored_data_dir, 'scores_no_trans.npy')
    save_label_score(score_path, relative_scores)
    print('saved!')
    
