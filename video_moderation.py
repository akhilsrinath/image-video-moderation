import cv2 
import math 
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.preprocessing import image 
import numpy as np 
from keras.utils import np_utils 
from skimage.transform import resize 
from skimage import io 
import math
from keras.preprocessing.image import img_to_array
from keras.models import load_model 
import numpy as np 
import argparse 
import imutils 
import cv2 
from collections import Counter
import os 

# vids = videos to be moderated

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
args = vars(ap.parse_args())


for videofile in vids:
	cap = cv2.VideoCapture(videofile)		# video object
	count = 0
	fps = cap.get(cv2.CAP_PROP_FPS)			# frames per second
	num_frames = []				
	ex_list = []			# list of explisit frames
	ap_list = []			# list of approved frames
	label_stat = []			# list of all labels of all frames 
	
	while cap.isOpened():
		ret, frame = cap.read()

		if ret:
			orig = frame.copy()			# copy of frame 
			frame = cv2.resize(frame, (28,28))	
			frame = frame.astype("float")/255.0
			frame = img_to_array(frame)		# convert frame to array
			frame = np.expand_dims(frame, axis=0)
			model = load_model(args["model"])			# load trained model
			(ok, notok) = model.predict(frame)[0]		# make prediction
			label = "Explicit" if notok>ok else "Approved"
			if label == "Explicit":
				ex_list.append(frame)		# update explicit list
			elif label == "Approved":
				ap_list.append(frame)		# update approved list
			label_stat.append(label)
			count += (3*fps)
			cap.set(1,count)
		else:
			cap.release()
			break

	c = Counter(label_stat)
	# print(videofile, "--->", c)
	l = sorted([(i, c[i]/len(label_stat)*100.0) for i in c])
	print(l)
	if l[0][1] == 100.0:
		print('Approved')
	elif (15.0 < l[1][1] < 50.0):
		print('Requires manual checking')
	elif (l[1][1]<15.0):
		print('Approved')
	elif (l[1][1] > 50.0):
		print('Explicit') 
	# print(len(ex_list))
	# print(len(ap_list))
