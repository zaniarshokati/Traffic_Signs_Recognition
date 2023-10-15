import tensorflow as tf 

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