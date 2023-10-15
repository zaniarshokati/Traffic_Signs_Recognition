import matplotlib.image as mpimg 
import os 

from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
# from keras.utils.np_utils import to_categorical 
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.optimizers import Adam 

from keras import layers 
from tensorflow import keras 
 
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from glob import glob 
import cv2 

import utilities

import warnings 
warnings.filterwarnings('ignore') 

class Application :
	def __init__(self) -> None:
		self.data_processor = utilities.ProcessData()
		self.visualize = utilities.Visualization()
		self.model_handler = utilities.HandleModel()
	def main(self):
		# path to the folder containing our dataset 
		dataset = 'data/traffic_Data/DATA'

		# path of label file 
		label_file = pd.read_csv('data/labels.csv') 

		train_ds,val_ds = self.data_processor.load_datasets(dataset)
		# self.visualize.show_sample_data(train_ds, label_file)
		input_shape = (224, 224, 3)
		model = self.model_handler.create_model(input_shape,label_file)
		

		# print(model.summary())

		# keras.utils.plot_model( 
		# 	model, 
		# 	show_shapes=True, 
		# 	show_dtype=True, 
		# 	show_layer_activations=True
		# ) 
		model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
					optimizer='adam', 
					metrics=['accuracy']) 
		# Set callback functions to early stop training 
		mycallbacks = [EarlyStopping(monitor='val_loss', patience=5)] 
		history = model.fit(train_ds, 
						validation_data=val_ds, 
						epochs=3, 
						callbacks=mycallbacks) 

		# Loss 
		plt.plot(history.history['loss']) 
		plt.plot(history.history['val_loss']) 
		plt.legend(['loss', 'val_loss'], loc='upper right') 

		# Accuracy 
		plt.plot(history.history['accuracy']) 
		plt.plot(history.history['val_accuracy']) 
		plt.legend(['accuracy', 'val_accuracy'], loc='upper right') 
		plt.show()

if __name__ == "__main__":
    app = Application()
    app.main()