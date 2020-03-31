##############################################################################################
# get roc-auc on image pairs
##############################################################################################
# import pdb
# from config import get_config
# import argparse
# from Learner import face_learner
# from data.data_pipe import get_val_pair
# from torchvision import transforms as trans
# from pathlib import Path
# from data.data_pipe import load_noonan_val_pair

# conf = get_config(training=False)
# learner = face_learner(conf, inference=True)
# learner.load_state(conf, 'mobilefacenet.pth', True, True)
# fp, issame = load_noonan_val_pair('data/facebank/noonan+normal/raw_112', 
#                                     Path('data/facebank/noonan+normal/val_112'), conf.test_transform)
# accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, fp, issame, nrof_folds=10, tta=True)
# print('fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
# trans.ToPILImage()(roc_curve_tensor)

##############################################################################################
# get landmarks from faces
##############################################################################################
# from data.data_pipe import img2lmk

# # img_path = 'data/facebank/webface/imgs'
# # lmk_path = 'data/facebank/webface/lmks'
# img_path = 'data/facebank/noonan+normal/raw_112'
# lmk_path = 'data/facebank/noonan+normal/lmks_112_raw'
# img2lmk(img_path, lmk_path, in_place=False)

##############################################################################################
# remove faces where landmarks cannot be detected
##############################################################################################
# from data.data_pipe import mv_no_face_img

# record = 'tmp'
# img_path = 'data/facebank/webface/imgs'
# no_face_path = 'data/facebank/webface/no_face'
# mv_no_face_img(record, img_path, no_face_path)

##############################################################################################
# change lanmarks according to noonan faces
##############################################################################################
# from data.data_pipe import cg_lmk

# img_path = 'data/facebank/noonan+normal/resize_112'
# lmk_path = 'data/facebank/noonan+normal/cg_112_resize'
# cg_lmk(img_path, lmk_path)

##############################################################################################
# get fake images that can be recognized by mtcnn
##############################################################################################
# from data.data_pipe import get_vague_faces

# records = ['raw_fr_lag_aug_pix2pix_transfer_b6_100_2layer', 'fr_lag_aug_pix2pix_transfer_b6_100_2layer',
#             'fr_lag_pix2pix_transfer_b6_100_2layer', 'fr_lag_aug_pix2pix_transfer_b6_500_2layer', 
#             'fr_mix_aug_pix2pix_transfer_b6_500_DG', 'fr_mix_pix2pix_transfer_b6_500_DG', 
#             'fr_mix_aug_pix2pix_transfer_b6_2000_DG', 'fr_mix_pix2pix_transfer_b6_2000_DG',
#             'fr_mix_aug_pix2pix_transfer_b6_100_pool5_full', 'fr_mix_aug_pix2pix_transfer_b6_100_full',
#             'fr_mix_aug_pix2pix_transfer_b6_2000_D2G', 'fr_mix_pix2pix_transfer_b6_2000_D2G']
# # for rec in records:
# # rec = records[-1]
# new_recs = records[-3:]
# for rec in new_recs:
#     print('processing', rec)
#     source_path = '../pytorch-CycleGAN-and-pix2pix/results/' + rec + '/test_latest/images'
#     save_path = 'data/facebank/noonan+normal/fake_' + rec
#     get_vague_faces(source_path, save_path)

# source_path = '../pytorch-CycleGAN-and-pix2pix/results/fr_aug_pix2pix_transfer_b6_100_2layer/test_latest/images'
# save_path = 'data/facebank/noonan+normal/fake_fr_aug_pix2pix_transfer_b6_100_2layer'
# get_vague_faces(source_path, save_path)

# sources = ['raw', 'resize']
# epochs = ['20', 'latest']
# # sources = ['raw_15', 'resize_15', 'raw_resize_15']
# models = ['fr_adult_pix2pix_transfer_b6_25_1layer_fe', 'fr_adult_pix2pix_transfer_b6_25_2layer',
#           'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG']

# for e in epochs:
#     for source in sources:
#         for model in models:
#             print('processing ' + source + '_' + e + ' on ' + model + '...')
#             source_path = ('../pytorch-CycleGAN-and-pix2pix/results/' + source + '_' + e + '/' + 
#                             model + '/test_' + e + '/images')
#             save_path = 'data/facebank/noonan+normal/fake_' + source + '_' + e + '_' + model
#             get_vague_faces(source_path, save_path)

##############################################################################################
# generate images for data augmentation in gan
##############################################################################################
from data.data_pipe import get_train_dataset_gan, img2lmk

print('get images from mtcnn...')
imgs_folder = 'data/facebank/divided/orig'
# imgs_folder = 'data/facebank/distinct/orig'
# imgs_folder = 'data/facebank/LAG_y_fine/orig'
# imgs_folder = 'data/facebank/noonan+normal/raw'
target_size = 112 #+ 5
target_folder = 'data/facebank/divided/mtcnn_' + str(target_size)
# target_folder = 'data/facebank/distinct/mtcnn_' + str(target_size)
# target_folder = 'data/facebank/LAG_y_fine/mtcnn_' + str(target_size)
# target_folder = 'data/facebank/noonan+normal/mtcnn_' + str(target_size)
get_train_dataset_gan(imgs_folder, target_folder, (target_size, target_size))

# print('get landmarks...')
# img_path = target_folder
# lmk_path = 'data/facebank/divided/lmks_mtcnn_' + str(target_size)
# # lmk_path = 'data/facebank/distinct/lmks_mtcnn_' + str(target_size)
# # lmk_path = 'data/facebank/LAG_y_fine/lmks_mtcnn_' + str(target_size)
# # lmk_path = 'data/facebank/noonan+normal/lmks_mtcnn_' + str(target_size)
# img2lmk(img_path, lmk_path, in_place=False)

##############################################################################################
# get_lag_y_data
##############################################################################################
# from data.data_pipe import get_lag_y_data

# lag_data = 'data/facebank/LAGdataset_200'
# lag_y_data = 'data/facebank/LAG_y'
# get_lag_y_data(lag_data, lag_y_data)

##############################################################################################
# merge A B as aligned data
##############################################################################################
# import os
# from PIL import Image

# Adir = 'data/facebank/LAG_y_fine/trainA'
# Bdir = 'data/facebank/LAG_y_fine/trainB'
# ABdir = 'data/facebank/LAG_y_fine/train'
# os.makedirs(ABdir)
# for name in os.listdir(Adir):
#     a = Image.open(Adir + os.sep + name)
#     b = Image.open(Bdir + os.sep + name)
#     ab = Image.new(a.mode, (a.size[0] * 2, a.size[1]))
#     ab.paste(a, box=(0,0))
#     ab.paste(b, box=(a.size[0],0))
#     ab.save(ABdir + os.sep + name)
##############################################################################################
# get lmks as np array
##############################################################################################
# from data.data_pipe import img2lmk_np

# img_path = 'data/facebank/distinct/raw_112'
# lmk_path = 'data/facebank/distinct/npy'
# img2lmk_np(img_path, lmk_path)