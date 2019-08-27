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
import heapq
#{'dad': 0, 'dau': 1, 'mom': 2, 'son': 3}

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

list_preds = []
dad_preds = []
mom_preds = []
model = load_model('../models/fam_class_model.h5')
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
        pred = model.predict(img_for_pred)[0]
        list_preds.append(pred)
        dad_preds.append(pred[0])
        mom_preds.append(pred[2])

        # pred_int = np.argmax(pred, axis=1)
        # if pred_int == 0:
        #     print('dad', file, pred)
        # elif pred_int == 1:
        #     print('dau', file, pred)
        # elif pred_int == 2:
        #     print('mom', file, pred)
        # elif pred_int == 3:
        #     print('son', file, pred)

dad = dad_preds.index(max(dad_preds))
mom = mom_preds.index(max(mom_preds))
if mom == dad:
    if mom_preds[mom] > dad_preds[dad]:
        dad = heapq.nlargest(2, range(len(dad_preds)), key=dad_preds.__getitem__)[1]
    elif mom_preds[mom] < dad_preds[dad]:
        mom = heapq.nlargest(2, range(len(mom_preds)), key=mom_preds.__getitem__)[1]
dau_list = []
son_list = []
for idx, i in enumerate(list_preds):
    if idx != dad and idx != mom:
        if i[1] > i[3]:
            dau_list.append(idx)
        elif i[1] > i[3]:
            son_list.append(idx)


print("pic", dad, "is dad", list_preds[dad][0], onlyfiles[dad])
print("pic", mom, "is mom", list_preds[mom][2], onlyfiles[mom])
for i in dau_list:
    print("pic", i, "is a dau", list_preds[i][1], onlyfiles[i])
for i in son_list:
    print("pic", i, "is a son", list_preds[i][3], onlyfiles[i])
