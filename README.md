# Cat-Dog Classifier

A machine learning project that uses Convolutional Neural Networks (CNN) to classify images as either cats or dogs. This project is implemented with TensorFlow and Keras for the model, and Streamlit for the web application interface.

![Cat-Dog Classifier GIF](https://github.com/saadtariq10/cat-dog-classifier/blob/main/cat-dog.gif?raw=true)

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How to Run the Application](#how-to-run-the-application)
- [Results](#results)

## Project Overview

The Cat-Dog Classifier is designed to classify images into two categories: cats and dogs. The model is trained on a dataset of labeled images, and users can interact with the model through a simple web interface built with Streamlit.

## Technologies Used

- **Python**: The programming language used for implementing the model and web app.
- **TensorFlow**: For building and training the CNN model.
- **Keras**: High-level neural networks API, running on top of TensorFlow.
- **Streamlit**: For creating the web application interface.
- **OpenCV**: For image processing.

## Dataset

The dataset used in this project is the [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset from Kaggle, containing images of cats and dogs. The dataset is organized into training and testing folders, with separate directories for each class.

## Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of the following layers:
- **Convolutional Layers**: Extract features from images.
- **Batch Normalization**: Normalizes the output of the previous layer to speed up training.
- **Max Pooling Layers**: Reduces the spatial dimensions of the output volume.
- **Flatten Layer**: Converts the 2D matrix into a 1D vector.
- **Dense Layers**: Fully connected layers for classification.
- **Output Layer**: Uses a sigmoid activation function for binary classification.

## How to Run the Application

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/saadtariq10/cat-dog-classifier.git
   cd cat-dog-classifier
