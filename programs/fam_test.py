from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import math
import natsort
from keras.models import load_model
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import numpy as np
from skimage import transform
from collections import Counter
import cv2

#{'dad': 0, 'dau': 1, 'mom': 2, 'son': 3}

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

model = load_model('/Users/brycekroencke/Documents/GitHub/family_classifier/models/fam_class_model.h5')
path_to_imgs = '../test'
onlyfiles = [f for f in listdir(path_to_imgs) if isfile(join(path_to_imgs, f))]
os.chdir(path_to_imgs)
onlyfiles=natsort.natsorted(onlyfiles)
if onlyfiles[0].startswith('.'):
    onlyfiles.remove(onlyfiles[0])
for id, file in enumerate(onlyfiles):
    if not file.startswith('.'):
        path = os.path.expanduser(file)
        img_for_pred = load(path)
        pred = model.predict(img_for_pred)
        pred_int = np.argmax(pred, axis=1)
        if pred_int == 0:
            print('dad', file, pred)
        elif pred_int == 1:
            print('dau', file, pred)
        elif pred_int == 2:
            print('mom', file, pred)
        elif pred_int == 3:
            print('son', file, pred)
