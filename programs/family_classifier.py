import time
import os
import sys
import sklearn
import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import backend as K
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.utils import to_categorical
import pylab as plt
from numpy.random import seed
from keras.optimizers import Adam
from tensorflow import set_random_seed

class hypers:
    num_classes = 4
    img_width = 64
    img_height = 64
    batch_size = 32
    lr = 1e-3
    epochs = 50
    loss = 'binary_crossentropy'


def main():
    #set seeds
    seed(1)
    set_random_seed(2)

    """
    CHECK FOR DEV MODE
    DEV MODE: less epochs
    """
    DEV = False
    argvs = sys.argv
    argc = len(argvs)
    if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
      DEV = True

    #make p a class containing desired hyperparameters
    p = hypers()

    if DEV:
        #in development mode
        p.epochs = 3

    #begin clocking runtime
    start = time.time()

    trained_model = train_model(p)
    save_trained_model(trained_model)
    # model_testing(trained_model, p)

    print_report(p, start)


"""
RETRIEVE DATA FROM DIRECTORIES AND PREPROCESS
"""
def pull_and_preprocess_train_data(p):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        directory=r"../reorganized_dset",
        target_size=(p.img_width, p.img_height),
        color_mode="rgb",
        batch_size=p.batch_size,
        shuffle=True,
        class_mode="categorical"
    )
    #
    # valid_generator = test_datagen.flow_from_directory(
    #     r"/Users/brycekroencke/Documents/ORNL/hayleys_dataset/Validation",
    #     target_size=(p.img_width, p.img_height),
    #     color_mode="rgb",
    #     batch_size=p.batch_size,
    #     shuffle=True,
    #     class_mode="categorical"
    # )


    return train_generator#, valid_generator

def pull_and_preprocess_test_data(p):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        r"/Users/brycekroencke/Documents/ORNL/hayleys_dataset/Test",
        target_size=(p.img_width, p.img_height),
        color_mode="rgb",
        batch_size=1,
        shuffle=False,
        class_mode="binary"
    )
    return test_generator



"""
CONSTRUCT MODEL
"""
def get_model(p):
    model = Sequential()
    chanDim = -1

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

	# (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
	# first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    opt = Adam(lr=p.lr, decay=p.lr / p.epochs)
    model.add(Dense(4, activation = 'softmax'))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=p.loss, optimizer=opt, metrics = ['accuracy'])
    print(model.summary())

    return model

"""
START TRAINING
"""
def train_model(p):
    train_generator = pull_and_preprocess_train_data(p)#, valid_generator = pull_and_preprocess_train_data(p)
    # for i in train_generator:
    #     idx = (train_generator.batch_index - 1) * train_generator.batch_size
    #     print(train_generator.filenames[idx : idx + train_generator.batch_size])
    #     print(train_generator.class_indices)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    #STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    model = get_model(p)
    model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=p.epochs#,
        # validation_data=valid_generator,
        # validation_steps=STEP_SIZE_VALID
    )

    # """
    # EVALUATING RESULTS
    # """
    # model.evaluate_generator(
    #     generator=valid_generator,
    #     steps=STEP_SIZE_VALID
    # )
    return model


"""
PREDICT ON TEST SET
"""
def model_testing(model, p):
    test_generator = pull_and_preprocess_test_data(p)
    test_generator.reset()
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    pred=model.predict_generator(
        test_generator,
        STEP_SIZE_TEST,
        verbose = 1
    )



def save_trained_model(model):
    model.save('../models/fam_class_model.h5')
    model.save_weights('../models/fam_class_weights.h5')


"""
Print total runtime of training and hyperparameters used
"""
def print_report(p, start):
    end = time.time()
    dur = end-start
    if dur<60:
        print("Execution Time:",dur,"seconds")
    elif dur>60 and dur<3600:
        dur=dur/60
        print("Execution Time:",dur,"minutes")
    else:
        dur=dur/(60*60)
        print("Execution Time:",dur,"hours")
    print("Img Dimensions:", p.img_width, "X", p.img_height)
    print("Epochs:", p.epochs)
    print("Batch Size:", p.batch_size)
    print("Optimizer:", p.opt)
    print("Loss:", p.loss)


if __name__== "__main__":
  main()
