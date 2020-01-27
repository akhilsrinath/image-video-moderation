# import libraries

import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical 
from pyimage.lenet import LeNet 		# Neural network model 
from imutils import paths 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse 
import random 
import cv2 
import os 

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help='path to input dataset')
ap.add_argument("-m", "--model", required=True, help='path to output model')
args = vars(ap.parse_args())

EPOCHS = 25
INIT_LR = 1e-3
BS = 32

print("[INFO] loading images...")
data = []
labels = [] 

imagePaths = sorted(list(paths.list_images(args['dataset'])))		# paths to input images
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	try:
		image = cv2.imread(imagePath)    # read image
		image = cv2.resize(image, (28,28))	
		# print(imagePath)
		image = img_to_array(image)		# convert image to array
		data.append(image)

		label = imagePath.split(os.path.sep)[-2]
		label = 1 if label == 'weapon' else 0
		labels.append(label)
	except:
		continue

data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=0) # split into training and test sets

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])














