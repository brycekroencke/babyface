import cv2
import sys
import os
from os import listdir
from PIL import Image
from os.path import isfile, join
import math
import natsort
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import numpy as np
from skimage import transform
from keras.models import load_model
import heapq


def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image





model = load_model('../models/fam_class_model.h5')
# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for idx, (x, y, w, h) in enumerate(faces):
    crop_img = image[y:y+h, x:x+w]
    cv2.resize(crop_img, (64, 64))
    cv2.imshow("cropped"+str(idx), crop_img)
    cv2.imwrite("../face_detect_made_dset/"+str(idx)+'.png',crop_img)


for idx, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)

path_to_imgs = '../face_detect_made_dset'
onlyfiles = [f for f in listdir(path_to_imgs) if isfile(join(path_to_imgs, f))]
os.chdir(path_to_imgs)
onlyfiles=natsort.natsorted(onlyfiles)
if onlyfiles[0].startswith('.'):
    onlyfiles.remove(onlyfiles[0])


list_preds = []
dad_preds = []
mom_preds = []

for id, file in enumerate(onlyfiles):
    if not file.startswith('.'):
        path = os.path.expanduser(file)
        img_for_pred = load(path)
        pred = model.predict(img_for_pred)[0]
        print(file, pred)
        list_preds.append(pred)
        dad_preds.append(pred[0])
        mom_preds.append(pred[2])

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


#print("pic", dad, "is dad", list_preds[dad][0], onlyfiles[dad])
os.rename(onlyfiles[dad], "dad.png")
#print("pic", mom, "is mom", list_preds[mom][2], onlyfiles[mom])
os.rename(onlyfiles[mom], "mom.png")
for i in dau_list:
    os.rename(onlyfiles[i], "dau"+str(i)+".png")
    #print("pic", i, "is a dau", list_preds[i][1], onlyfiles[i])
for i in son_list:
    os.rename(onlyfiles[i], "son"+str(i)+".png")
    #print("pic", i, "is a son", list_preds[i][3], onlyfiles[i])

cv2.waitKey(0)
