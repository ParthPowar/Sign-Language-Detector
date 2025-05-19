
Sign Language Interpreter using Deep Learning
Real-time American Sign Language (ASL) interpreter using live webcam feed.

Overview
Developed as part of HACK UNT 19, the aim of this project is to enhance accessibility by empowering individuals with hearing disabilities. With over 70 million deaf people globally, we sought to build a personal sign language interpreter that operates continuously—allowing users to communicate independently without needing a human translator.

Technologies Used
Python

TensorFlow

Keras

OpenCV

Setup Instructions
Clone the repository and navigate to the project directory.

Open a command prompt and install the required packages using one of the provided text files:

bash
Copy
Edit
python -m pip install -r install_packages.txt
or for GPU support:

bash
Copy
Edit
python -m pip install -r install_packages_gpu.txt
This will install all the dependencies required for the project.

How It Works
Run set_hand_histogram.py to generate a hand histogram used for gesture segmentation.

After obtaining a satisfactory histogram, save it in the code directory. Alternatively, you can use our pre-generated histogram available here.

Create gesture datasets using your webcam by running create_gestures.py. This script captures hand signs and stores them with corresponding labels.

Enhance dataset diversity by flipping images using Rotate_images.py.

Run load_images.py to divide the dataset into training, validation, and testing sets.

Use display_gestures.py to preview your collected gestures.

Train the Convolutional Neural Network (CNN) model using cnn_model_train.py.

Finally, launch final.py to start real-time sign recognition through your webcam.

Code Example
python
Copy
Edit
# Model Training using CNN

import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(glob('gestures/*'))

image_x, image_y = get_image_size()

def cnn_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()
	model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="cnn_model_keras2.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("val_images", "rb") as f:
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels)
	val_labels = np_utils.to_categorical(val_labels)

	print(val_labels.shape)

	model, callbacks_list = cnn_model()
	model.summary()
	model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)
	scores = model.evaluate(val_images, val_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))

train()
K.clear_session();
Features
Recognizes 44 ASL gestures with over 95% accuracy.

Trained using a custom CNN model tailored for real-time recognition.

Works directly with live webcam feed for gesture interpretation.

Future Enhancements
Deploy as a cloud-based API to support integration into third-party apps.

Expand gesture vocabulary to cover full ASL and other sign languages.

Implement a feedback loop to improve model performance with user input.

Add multilingual or region-specific sign language support.

