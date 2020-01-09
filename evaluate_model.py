import pdb
from config import get_config
import argparse
from Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
from pathlib import Path
from data.data_pipe import load_noonan_val_pair

conf = get_config(training=False)
learner = face_learner(conf, inference=True)
learner.load_state(conf, 'mobilefacenet.pth', True, True)
fp, issame = load_noonan_val_pair('data/facebank/noonan+normal/raw_112', Path('data/facebank/noonan+normal/val_112'), conf.test_transform)
accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, fp, issame, nrof_folds=10, tta=True)
print('fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
trans.ToPILImage()(roc_curve_tensor)
