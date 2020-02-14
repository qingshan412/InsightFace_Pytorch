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
from data.data_pipe import img2lmk

# img_path = 'data/facebank/nweface/imgs'
# lmk_path = 'data/facebank/webface/lmks'
img_path = 'data/facebank/noonan+normal/resize_112'
lmk_path = 'data/facebank/noonan+normal/lmks_112'
img2lmk(img_path, lmk_path, in_place=False)

##############################################################################################
# remove faces where landmarks cannot be detected
##############################################################################################
# from data.data_pipe import mv_no_face_img

# record = 'tmp'
# img_path = 'data/facebank/webface/imgs'
# no_face_path = 'data/facebank/webface/no_face'
# mv_no_face_img(record, img_path, no_face_path)

##############################################################################################
# resize to 112x112 without mtcnn
##############################################################################################
# import os
# from PIL import Image

# raw_dir = 'data/facebank/noonan+normal/raw'
# resize_dir = 'data/facebank/noonan+normal/resize_112'
# os.makedirs(resize_dir, exist_ok=True)

# for f in os.listdir(raw_dir):
#     img = Image.open(raw_dir + os.sep + f)
#     sz = img.size
#     if sz[0] > sz[1]:
#         diff = sz[0] - sz[1]
#         left = diff // 2
#         img.crop((left, 0, sz[0] - (diff - diff // 2), sz[1])).resize((112, 112)).save(resize_dir + os.sep + f)
#     elif sz[0] < sz[1]:
#         diff = sz[1] - sz[0]
#         # high = diff // 2
#         img.crop((0, diff, sz[0], sz[1])).resize((112, 112)).save(resize_dir + os.sep + f)
#     # resize_img = Image.open(raw_dir + os.sep + f).resize((112, 112)).save(resize_dir + os.sep + f)