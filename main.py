from scripts.satellite_images import SatelliteImages
from scripts.dataset import CustomDataset
from scripts.model import torch, Model, TransformerModel
from scripts.train_data import COORDINATES
from scripts.helpers import PROJECT, TORCH_MODEL_PATH, os, DataLoader, nn, np, optim, SummaryWriter, \
    train_validate_model, test_model, random_split, visualize_predictions, compute_pos_weight, BCEDiceLoss, \
    train_test_split, model_progress, select_device

'''
Main function to run the project.
- create functions for:
    - generating training images
    - training model
    - generating unseen images

'''

def main():

    # initialize objects
    # initialize Images Object
    images = SatelliteImages()
    # instanciate processor class
    p = CustomDataset()

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
        SatelliteImages.initializer(PROJECT)

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

    # generate unseen images
    if unseen_input in ['y', 'yes']:
        # try generate 5 new images
        # if image dont exist at earth engine
        # one less image will be generated
        images.create_unseen_data(5)


if __name__ == '__main__':
    # use random seed for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    device = select_device()
    # main()
    model = TransformerModel().to(device) # instantiate model and send to device

    dataset = CustomDataset()  # instantiate the dataset
    dataset.load_train_data()  # load the training data

    # hyperparameters
    batch_size = 2
    learning_rate = 2e-4
    num_epochs = 10

    # split Data
    num_samples = len(dataset)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_loader, device)
    pos_weight = torch.clamp(pos_weight, max=10.0)
    pos_weight = pos_weight.clone().detach().to(device)

    criterion = BCEDiceLoss(pos_weight=pos_weight, dice_weight=1.0, bce_weight=1.0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # define writer for tensorboard logging
    writer = SummaryWriter(f"runs/{PROJECT}")

    # Train or load model
    if os.path.exists(TORCH_MODEL_PATH):
        print("Loading existing model...")
        model.load_state_dict(torch.load(TORCH_MODEL_PATH, weights_only=True))
    else:
        print("Training new model...")
        train_losses, val_losses, model, writer, epoch = train_validate_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
            writer,
            patience=3
        )

    #close the tensorboard writer
    writer.flush()
    writer.close()

    # see predictions
    visualize_predictions(model, test_loader, device, max_images=test_size, threshold=0.5)
