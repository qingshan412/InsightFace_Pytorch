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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-ds", "--dataset_dir", help="where to get data", default="noonan+normal", type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
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
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    test_dir = conf.data_path/'facebank'/args.dataset_dir/'test'
    if args.tta:
        verify_type = args.dataset_dir + '_tta'
    else:
        verify_type = args.dataset_dir
    verify_dir = conf.data_path/'facebank'/verify_type/'verify'
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
                    try:
                        image = Image.open(fil)
                    except:
                        continue
                    
                    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                    bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1,-1,1,1] # personal choice    
                    results, score = learner.infer(conf, faces, targets, args.tta)
                    for idx,bbox in enumerate(bboxes):
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    
                    cv2.imshow(fil.name, frame)
                    cv2.imwrite(verify_dir/fil.name, frame)


