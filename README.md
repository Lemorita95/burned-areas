# BURNED AREAS
convolutional neural network to identify burned areas from png satellite images.

model was trained with sentinel2 surface reflectance (S2_SR) images.

## content
[requirements.txt](requirements.txt) -  python libraries requirement

[main.py](main.py) - main file

[helpers.py](helpers.py) - auxiliary functions

[coordinates.py](coordinates.py) - holds train data coordinates at earth engine

[image_processor.py](image_processor.py) - class to process image to feed the model

[make_predictions.py](make_predictions.py) - predictions at unseen data

[model.py](model.py) - CNN model declaration

[model.keras](model.keras) - model export

[satellite_images.py](satellite_images.py) - class to handle satellite image aquisition

## description
data was acquired with S2_SR images where burned areas was identified with the help of Normal difference vegetation index (NDVI), normal difference water index (NDWI), near infrared band, and normalized burn ratio (NBR).

### coordinates.py
this file contains information of earth engine train data of the model.

### helpers.py
this file contains global variables, global paths and utile functions used across multiple files and all library imports.

### image_processor.py
this file contains the Processor() object. it handles how the .png image is acquires from a specific folder and loaded into the program in the desired format.

### main.py
this file contains the executable for training image aquisition, model training and unseen image acquisition.

### make_predictions.py
in this file, the unseen image is inputted at a trained model and outputs the predicted image, i.e. detections of burned area in the original image.

### model.keras
trained model.

### model.py
this file holds the Model() object, i.e. the Convolutional Neural Network with its architecture.

### satellite_images.py
this file deals with the interface with earth engine, the aquisiton of image collection, binary mask, image save.

## to-do
data is unbalance. implement things to deal with it for better predictions.