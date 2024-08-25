import os
import datetime

# satellite_images.py
import requests
from PIL import Image
from io import BytesIO
import ee
import random

# make_predictions.py and image_processor.py
import cv2
import numpy as np
from copy import deepcopy

# model.py and main.py
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
UNSEEN_DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unseen_data')
PREDICTIONS_DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'predictions')

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.keras')

# visualization parameters
VIS_PARAMETERS = {
    'bands': ['B4', 'B3', 'B2'],  # rgb
    'min': 0,
    'max': 3000,
    'gamma': 1.4,
}

# resolution when generating image thumb to save as png
PNG_RESOLUTION = (480, 480, 3)

IMG_WIDTH = 480
IMG_HEIGHT = 480


def to_timestamp(value_in_ms):
    """"
    function to format ee.image generation date into a timestamp format
    """
    return datetime.datetime.fromtimestamp(value_in_ms // 1000)


def save_image(img, folder, filename):
    """ 
    this function save each set of images in a separate folder
    \nit is used to assure that evicende and label will match
    """
    
    # Create the full directory path if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Combine the folder and filename to create the full file path
    file_path = os.path.join(folder, filename)
    
    # Save the image
    img.save(file_path)


def model_progress(history):

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.show()