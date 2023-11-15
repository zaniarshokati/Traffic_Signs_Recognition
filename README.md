# Traffic Signs Recognition

This project aims to recognize and classify traffic signs using Convolutional Neural Networks (CNN) and Keras in Python. 
## Overview

The code is organized into two main files:

- **main.py**: This file contains the main application class (`Application`) responsible for coordinating the data processing, model handling, and visualization. It utilizes the utility functions defined in `utils.py` to load datasets, create, compile, and train the CNN model, as well as visualize sample data and training history.

- **utils.py**: This file encapsulates utility classes for data processing (`ProcessData`), visualization (`Visualization`), and model handling (`HandleModel`). The CNN model is defined using Keras, incorporating data augmentation techniques, convolutional layers, max-pooling layers, and dense layers.

## Usage

1. **Download Dataset**: To reproduce the results and use this code, download the traffic signs dataset from [Kaggle](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification/code).

2. **Set Up Environment**: Ensure you have the required dependencies installed. You can install them using:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Code**: Execute `main.py` to load the dataset, create and train the model, and visualize the results.

    ```bash
    python main.py
    ```

## Results

The project incorporates data augmentation, CNN architecture, and early stopping to achieve accurate traffic sign recognition. The visualization module provides insights into the training process, including sample data display and loss-accuracy plots.

For more details on the project and its implementation, refer to the source code and documentation.

## Acknowledgments

This project is inspired by the work described in [GeeksforGeeks - Traffic Signs Recognition using CNN and Keras in Python](https://www.geeksforgeeks.org/traffic-signs-recognition-using-cnn-and-keras-in-python/).
