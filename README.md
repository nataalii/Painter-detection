## Painter Detection


This project involves developing a Convolutional Neural Network (CNN) model to classify paintings based on the artist. The dataset consists of paintings from various artists, and the model aims to accurately predict the artist for each painting.


#

### Dataset
The dataset is organized in the directory ./paintings_dataset/images. Each subdirectory within this directory represents a different painter and contains the respective paintings by that artist.

#

### Installation

1\. First of all you need to clone repository from github:

```
git clone https://github.com/nataalii/painter-detection
```

2\. Installation
To run this project, you need to have Python installed along with several libraries. You can install the required libraries using the following command:

```
pip install torch torchvision torchmetrics scikit-learn matplotlib Pillow
```


# 

### Model Architecture

The model is a Convolutional Neural Network (CNN) designed to capture intricate details and patterns within the paintings. It includes:

Three sets of convolutional and max-pooling layers for feature extraction.
Fully connected layers for classification.
Training and Evaluation
The training process involves:

Loading and preprocessing the data.
Training the model over 35 epochs with an AdamW optimizer and Cross-Entropy Loss.
Evaluating the model using accuracy, precision, recall, and F1-score metrics.
Generating and plotting a confusion matrix to visualize performance.
Example Prediction
The project includes an example prediction function to classify a new painting. Use the predict_painter function to predict the artist of a given painting image.

#
### Results
The model is evaluated on a test dataset, providing detailed performance metrics and visualizations. The results demonstrate the model's effectiveness in classifying paintings by different artists.

### Contributing
Contributions to the project are welcome. If you have any suggestions or improvements, feel free to submit a pull request or open an issue.

### License
This project is licensed under the MIT License.

### Additional Information
Directory Structure
main.py: The main script to run the training, evaluation, and prediction processes.
src/data_preprocessing.py: Functions for loading and splitting the dataset.
src/model_training.py: The CNN model architecture and training loops.
src/model_evaluation.py: Functions for evaluating the model and plotting results.
src/predict.py: Function for making predictions on new images.