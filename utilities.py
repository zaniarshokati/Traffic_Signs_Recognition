import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping 

class ProcessData:
    def load_datasets(self,dataset):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.2, 
                                                            subset='training', 
                                                            image_size=(224, 224), 
                                                            seed=123, 
                                                            batch_size=32) 
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.2, 
                                                                    subset='validation', 
                                                                    image_size=(224, 224), 
                                                                    seed=123, 
                                                                    batch_size=32)

        return train_ds, val_ds
    
class Visualization:
    def __init__(self) -> None:
        pass

    def show_sample_data(self, train_ds,label_file):
        class_numbers = train_ds.class_names 
        class_names = [] 
        for i in class_numbers: 
            class_names.append(label_file['Name'][int(i)]) 

            
        plt.figure(figsize=(10, 10)) 
        for images, labels in train_ds.take(1): 
            for i in range(25): 
                ax = plt.subplot(5, 5, i + 1) 
                plt.imshow(images[i].numpy().astype("uint8")) 
                plt.title(class_names[labels[i]]) 
                plt.axis("off") 

        plt.show() 

class HandleModel:
    def __init__(self) -> None:
        pass

    def create_model(self,input_shape,label_file):
        data_augmentation = tf.keras.Sequential( 
			[ 
				tf.keras.layers.experimental.preprocessing.RandomFlip( 
					"horizontal", input_shape=input_shape), 
				tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), 
				tf.keras.layers.experimental.preprocessing.RandomZoom(0.2), 
				tf.keras.layers.experimental.preprocessing.RandomFlip( 
					mode="horizontal_and_vertical") 
			] 
		) 

        model = Sequential() 
        model.add(data_augmentation) 
        model.add(Rescaling(1./255)) 
        model.add(Conv2D(128, (3, 3), activation='relu')) 
        model.add(MaxPooling2D((2, 2))) 
        model.add(Conv2D(64, (3, 3), activation='relu')) 
        model.add(MaxPooling2D((2, 2))) 
        model.add(Conv2D(128, (3, 3), activation='relu')) 
        model.add(MaxPooling2D((2, 2))) 
        model.add(Conv2D(256, (3, 3), activation='relu')) 
        model.add(MaxPooling2D((2, 2))) 
        model.add(Flatten()) 
        model.add(Dense(64, activation='relu')) 
        model.add(Dropout(0.2)) 
        model.add(Dense(128, activation='relu')) 
        model.add(Dense(len(label_file), activation='softmax')) 

        return model
    
    def compile_model(self, model):
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                            optimizer='adam', 
                            metrics=['accuracy'])
        
        return model

    def train_model(self, model,train_ds,val_ds):
        mycallbacks = [EarlyStopping(monitor='val_loss', patience=5)] 
        history = model.fit(train_ds, 
                        validation_data=val_ds, 
                        epochs=3, 
                        callbacks=mycallbacks)
        
        return history