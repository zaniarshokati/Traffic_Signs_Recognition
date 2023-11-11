import matplotlib.pyplot as plt
import pandas as pd

import utils
import warnings

warnings.filterwarnings("ignore")


class Application:
    def __init__(self, data_path, label_file) -> None:
        self.data_processor = utils.ProcessData()
        self.visualize = utils.Visualization()
        self.model_handler = utils.HandleModel()
        self.data_path = data_path
        self.label_file = label_file
    def main(self): 
        
        train_ds, val_ds = self.data_processor.load_datasets(self.data_path)
        # self.visualize.show_sample_data(train_ds, label_file)
        input_shape = (224, 224, 3)
        model = self.model_handler.create_model(input_shape, label_file)

        print(model.summary())

        model = self.model_handler.compile_model(model)

        # Set callback functions to early stop training
        history = self.model_handler.train_model(model, train_ds, val_ds)

        self.visualize.show_loss_accuracy(history)


if __name__ == "__main__":
    data_path =  "data/traffic_Data/DATA"
    label_file = pd.read_csv("data/labels.csv")
    app = Application(data_path, label_file)
    app.main()
