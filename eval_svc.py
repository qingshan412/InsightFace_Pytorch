from PIL import Image
import argparse
from pathlib import Path
from config import get_config

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve
import os
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
    fp_tp = {}
    accuracy = {}
    for name in names_considered:
        fp_tp[name] = [[], []] # fpr_list, tpr_list
        accuracy[name] = []
    
    # prepare folders
    raw_dir = 'raw_112'
    verify_type = 'svm_' + args.kernel
    if args.use_shuffled_kfold:
        verify_type += '_shuffled'
    save_dir = conf.facebank_path/args.dataset_dir/verify_type
    os.makedirs(str(save_dir), exist_ok=True)
    print('folder ready...')

    # collect raw data
    data_dict = {}
    for name in names_considered:
        data_dict[name] = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/raw_dir) + '/' + name + '*'))

    # init kfold
    if args.use_shuffled_kfold:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=6)
    else:
        kf = KFold(n_splits=args.kfold, shuffle=False, random_state=None)
            
    for fold_idx, (train_index, test_index) in enumerate(kf.split(data_dict[names_considered[0]])):
        train_set = {}
        test_set = {}
        # sklearn style input
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for name in names_considered:
            train_set[name], test_set[name] = data_dict[name][train_index], data_dict[name][test_index]
            # transform images to numpy arrays
            print('train_set:')
            for i in train_set[name]:
                print(i)
                X_train.append(np.asarray(Image.open(i)))
                y_train.append(0 if names_considered[0] in i else 1) # binary
            print('test_set:')
            for i in test_set[name]:
                print(i)
                X_test.append(np.asarray(Image.open(i)))
                y_test.append(0 if names_considered[0] in i else 1) # binary
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(fold_idx)
        print('datasets ready...')

        # SVM classification
        clf = SVC(kernel=args.kernel,gamma='auto')
        clf.fit(X_train, y_train)

        # accury + ROC
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy:', accuracy)
        # plot_roc_curve(clf, X_test, y_test) # not yet available function
        pos_label = clf.classes_[1]
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        name = fold_idx + '_fold'
        line_kwargs = {
            'label': "{} (AUC = {:0.2f})".format(name, self.roc_auc)
        }
        line = ax.plot(fpr, tpr, **line_kwargs)[0]
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')
        plt.savefig(str(save_dir) + os.sep + fold_idx + '_fold.png')
