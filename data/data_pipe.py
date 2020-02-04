from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm

from itertools import combinations
from mtcnn import MTCNN
import os, glob, dlib, shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
        print('vgg loader generated')        
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder/'imgs')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num 
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def load_noonan_val_pair(path, rootdir, transform, image_size=[112,112]):
    # path: path for original images, str
    # rootdir: path to store images and issame_list, Path
    mtcnn = MTCNN()
    print('mtcnn loaded')

    if not rootdir.exists():
        rootdir.mkdir()
    images = os.listdir(path)
    comb_size = len(images) * (len(images) - 1) / 2
    pairs = combinations(images, 2)
    data = bcolz.fill([int(comb_size * 2), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    issame_list = np.zeros(int(comb_size))
    i = 0
    for pair in pairs:
        img0 = Image.open(os.path.join(path, pair[0]))
        if img0.size != (112, 112):
            img0 = mtcnn.align(img0)
        img1 = Image.open(os.path.join(path, pair[1]))
        if img1.size != (112, 112):
            img1 = mtcnn.align(img1)
        data[2*i, ...] = transform(img0)
        data[2*i + 1, ...] = transform(img1)
        if ('noonan' in pair[0] and 'noonan' in pair[1]) or ('normal' in pair[0] and 'normal' in pair[1]):
            issame_list[i] = 1
        i += 1
        if i % 1000 == 0:
            print('loading noonan', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)

### using system call for directory operatios instead of pathlib as before
def img2lmk(img_path, lmk_path, in_place=False, predictor_path='data/lmk_predictor/shape_predictor_68_face_landmarks.dat'):
    # os.makedirs(lmk_path, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    print('predictor loaded')
    
    if 'webface' in img_path:
        data_path = os.path.join(img_path, '*', '*.jpg')
    else:
        data_path = os.path.join(img_path, '*.jpg')
    for f in glob.glob(data_path):
        lmk_f = f.replace(img_path, lmk_path)
        # print(lmk_f)
        lmk_dir = os.sep.join(lmk_f.split(os.sep)[:-1])
        # print(lmk_dir)
        os.makedirs(lmk_dir, exist_ok=True)

        img = dlib.load_rgb_image(f)
        # Ask the detector to find the bounding boxes of each face. The 0 in the
        # second argument indicates that we should upsample the image 0 time. Usually, 
        # it's set to 1 to make everything bigger and allow us to detect more faces.
        dets = detector(img, 0)

        if len(dets) != 1:
            print("Processing file: {}".format(f))
            print("Number of faces detected: {}".format(len(dets)))
            continue
        rec = [dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()]
        print(rec)
        exit(0)
        
        shape = predictor(img, dets[0])
        points = [(p.x, p.y) for p in shape.parts()]

        if in_place:
            lmk_img = Image.open(f)
            plt.clf()
            plt.imshow(lmk_img)
            plt.scatter([p[0] for p in points], [p[1] for p in points], marker='o', color='r', linewidths=1)
            currentAxisA = plt.gca()
            rect = patches.Rectangle((rec[1], rec[0]), (rec[2] - rec[0]), (rec[3] - rec[1]), 
                                     linewidth=5, edgecolor='g', facecolor='none')
            currentAxisA.add_patch(rect)
            currentAxisA.axes.get_xaxis().set_visible(False)
            currentAxisA.axes.get_yaxis().set_visible(False)
            currentAxisA.spines['left'].set_color('none')
            currentAxisA.spines['bottom'].set_color('none')
            plt.savefig(lmk_f, bbox_inches='tight', pad_inches=0.0)
        else:
            lmk_img = np.ones(img.shape) * img.mean()
            lmk_img = Image.fromarray(lmk_img.astype('uint8'))
            lmk_draw = ImageDraw.Draw(lmk_img)
            lmk_draw.rectangle(rec, outline='green') #'black'
            lmk_draw.point(points, fill='red') #'white'
            del lmk_draw
            lmk_img.save(lmk_f)

### move images from which dlib cannot detect a face to a folder
def mv_no_face_img(record, orig_path, no_face_path):
    with open(record) as f:
        for line in f:
            if "Processing" in line:
                img_path = line.strip().split(' ')[-1]
                no_img_path = img_path.replace(orig_path, no_face_path)
                no_img_dir = os.sep.join(no_img_path.split(os.sep)[:-1])
                os.makedirs(no_img_dir, exist_ok=True)
                shutil.move(img_path, no_img_path)

# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label