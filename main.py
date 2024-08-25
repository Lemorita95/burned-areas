from satellite_images import SatelliteImages
from image_processor import Processor
from model import Model
from coordinates import COORDINATES

from helpers import *

def main():
    
    # project credentials
    project = "lnm-424214"

    # initialize objects
    # initialize Images Object
    images = SatelliteImages()
    # instanciate processor class
    p = Processor()

    # ask user input for generating images
    # input validation
    while True:
        generate_input = input("generate train images (y/n)? ")
        generate_input = generate_input.lower()

        if generate_input in ['y', 'yes', 'n', 'no']:
            break

    # generate images
    if generate_input in ['y', 'yes']:

        # initialize project
        SatelliteImages.initializer(project)

        # from coordinates.py file
        coordinates = COORDINATES

        # image acquisition and save
        images.create_train_data(coordinates)

    # ask user input for training model
    # input validation
    while True:
        train_input = input("create model (y/n)? ")
        train_input = train_input.lower()

        if train_input in ['y', 'yes', 'n', 'no']:
            break

    # train model and save
    if train_input in ['y', 'yes']:

        # Load and prepare data
        images, labels = p.load_train_data()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)

        # get model
        model = Model().model
        
        # Train the model on the training set
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

        model_progress(history)

        # Save model
        model.save('model.keras')
        print(f"Model saved to 'model.keras'.")

    # generate random unseen images
    while True:
        unseen_input = input("create unseen images (y/n)? ")
        unseen_input = unseen_input.lower()

        if unseen_input in ['y', 'yes', 'n', 'no']:
            break

    # train model and save
    if unseen_input in ['y', 'yes']:
        # try generate 5 new images
        # if image dont exist at earth engine
        # one less image will be generated
        images.create_unseen_data(5)


if __name__ == '__main__':
    main()