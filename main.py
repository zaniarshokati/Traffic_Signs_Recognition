import matplotlib.pyplot as plt 
import pandas as pd 

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
		

		print(model.summary())

		model = self.model_handler.compile_model(model)
		
		# Set callback functions to early stop training 
		history = self.model_handler.train_model(model,train_ds,val_ds)

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