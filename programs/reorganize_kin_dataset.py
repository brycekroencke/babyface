import os
from os import listdir
from os.path import isfile, join

mypath = '/Users/brycekroencke/Documents/GitHub/family_classifier/KinFaceW-II/images/father-dau'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in onlyfiles:
    fam_class = i.split('_')[-1].split('.')[0]
    if fam_class == str(1):
        print("dad")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/dad',i))
    elif fam_class == str(2):
        print("dau")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/dau', i))
    else:
        print("error")


mypath = '/Users/brycekroencke/Documents/GitHub/family_classifier/KinFaceW-II/images/father-son'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in onlyfiles:
    fam_class = i.split('_')[-1].split('.')[0]
    if fam_class == str(1):
        print("dad")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/dad',i))
    elif fam_class == str(2):
        print("son")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/son', i))
    else:
        print("error")


mypath = '/Users/brycekroencke/Documents/GitHub/family_classifier/KinFaceW-II/images/mother-son'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in onlyfiles:
    fam_class = i.split('_')[-1].split('.')[0]
    if fam_class == str(1):
        print("mother")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/mom',i))
    elif fam_class == str(2):
        print("son")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/son', i))
    else:
        print("error")


mypath = '/Users/brycekroencke/Documents/GitHub/family_classifier/KinFaceW-II/images/mother-dau'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in onlyfiles:
    fam_class = i.split('_')[-1].split('.')[0]
    if fam_class == str(1):
        print("mother")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/mom',i))
    elif fam_class == str(2):
        print("dau")
        os.rename(join(mypath, i), join('/Users/brycekroencke/Documents/GitHub/family_classifier/reorganized_dset/dau', i))
    else:
        print("error")
