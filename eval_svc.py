from PIL import Image
import argparse
from pathlib import Path
from config import get_config

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import os, glob
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for evaluation of svc')
    parser.add_argument("-ds", "--dataset_dir", help="where to get data", default="noonan+normal", type=str)
    parser.add_argument("-g", "--gpu_id", help="gpu id to use", default="", type=str)
    parser.add_argument("-k", "--kfold", help="returns the number of splitting iterations in the cross-validator.", 
                        default=10, type=int)
    parser.add_argument("-n", "--names_considered", help="names for different types considered, separated by commas", 
                        default="normal,noonan", type=str)
    parser.add_argument("-r", "--kernel", help="kernel name to use", default="rbf", type=str)
    parser.add_argument("-s", "--use_shuffled_kfold", help="whether to use shuffled kfold.", action="store_true")
    args = parser.parse_args()

    conf = get_config(False, args)
    print('config ready...')
    
    names_considered = args.names_considered.strip().split(',')
    accuracy = []
    # for name in names_considered:
    #     accuracy[name] = []
    
    # prepare folders
    raw_dir = 'npy' #'resize_112'
    verify_type = 'svm_' + args.kernel
    if args.use_shuffled_kfold:
        verify_type += '_shuffled'
    save_dir = conf.facebank_path/args.dataset_dir/verify_type
    os.makedirs(str(save_dir), exist_ok=True)
    print('folder ready...')

    # init kfold
    if args.use_shuffled_kfold:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=6)
    else:
        kf = KFold(n_splits=args.kfold, shuffle=False, random_state=None)

    # collect raw data
    data_dict = {}
    if 'npy' in raw_dir:
        label_dict = {'normal': 0, 'noonan': 1}
        names_npy = np.load(str(conf.data_path/'facebank'/args.dataset_dir/raw_dir/'img_names.npy'))
        labels_npy = np.array([1 if 'noonan' in item else 0 for item in names_npy])
        lmks_npy = np.load(str(conf.data_path/'facebank'/args.dataset_dir/raw_dir/'lmks.npy'))
        for name in names_considered:
            data_dict[name] = lmks_npy[np.where(labels_npy==label_dict[name])]
            idx_gen[name] = kf.split(data_dict[name])
    else:
        for name in names_considered:
            data_dict[name] = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/raw_dir) + 
                                        '/' + name + '*'))
            idx_gen[name] = kf.split(data_dict[name])
            
    for fold_idx in range(args.kfold):
        train_set = {}
        test_set = {}
        # sklearn style input
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for name in names_considered:
            (train_index, test_index) = next(idx_gen[name])
            train_set[name], test_set[name] = data_dict[name][train_index], data_dict[name][test_index]
            if 'npy' in raw_dir:
                # transform to numpy arrays
                for i in train_set[name]:
                    X_train.append(i.flatten())
                    y_train.append(1 if 'noonan' in name else 0) # binary
                for i in test_set[name]:
                    X_test.append(i.flatten())
                    y_test.append(1 if 'noonan' in name else 0) # binary
            else:
                # transform images to numpy arrays
                for i in train_set[name]:
                    X_train.append(np.asarray(Image.open(i)).flatten())
                    y_train.append(0 if names_considered[0] in i.strip().split(os.sep)[-1] else 1) # binary
                for i in test_set[name]:
                    X_test.append(np.asarray(Image.open(i)).flatten())
                    y_test.append(0 if names_considered[0] in i.strip().split(os.sep)[-1] else 1) # binary
        # print('train_set:', X_train, y_train)
        # print('test_set:', X_test, y_test)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(fold_idx)
        print('datasets ready...')

        # SVM classification
        clf = SVC(kernel=args.kernel,gamma='auto')
        # clf = MLPClassifier()
        clf.fit(X_train, y_train)

        # accury + ROC
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy.append(acc)
        print('accuracy:', acc)
        # plot_roc_curve(clf, X_test, y_test) # not yet available function
        pos_label = clf.classes_[1]
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        name = str(fold_idx) + '_fold'
        line_kwargs = {
            'label': "{} (AUC = {:0.2f})".format(name, roc_auc)
        }
        line = ax.plot(fpr, tpr, **line_kwargs)[0]
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')
        plt.savefig(str(save_dir) + os.sep + name + '.png')

    print('average accuracy:', np.mean(np.array(accuracy)))