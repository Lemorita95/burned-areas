import os
import datetime
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# satellite_images.py
import requests
from PIL import Image
from io import BytesIO
import ee
import random

# make_predictions.py and dataset.py
import cv2
import numpy as np
from copy import deepcopy
from torch.utils.data import random_split, Dataset, DataLoader
import albumentations as A

# model.py and main.py
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# environment variables
from dotenv import load_dotenv

MAIN_FOLDER = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IMAGES_FOLDER = os.path.join(MAIN_FOLDER, 'images')
UNSEEN_DATA_FOLDER = os.path.join(MAIN_FOLDER, 'unseen_data')
PREDICTIONS_DATA_FOLDER = os.path.join(MAIN_FOLDER, 'predictions')
MODEL_FOLDER = os.path.join(MAIN_FOLDER, 'model')

KERAS_MODEL_PATH = os.path.join(MODEL_FOLDER, 'model.keras')
TORCH_MODEL_PATH = os.path.join(MODEL_FOLDER, 'model.pth')

# visualization parameters
VIS_PARAMETERS = {
    'bands': ['B4', 'B3', 'B2'],  # rgb
    'min': 0,
    'max': 3000,
    'gamma': 1.4,
}

IMG_WIDTH = 480
IMG_HEIGHT = 480

# resolution when generating image thumb to save as png
PNG_RESOLUTION = (IMG_HEIGHT, IMG_WIDTH, 3)

load_dotenv(os.path.join(MAIN_FOLDER, '.env'))
PROJECT: str = os.getenv('PROJECT')

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


def select_device():
        # Check for available devices and select if available
    if torch.cuda.is_available():
        device = torch.device("cuda")       #CUDA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")        #Apple GPU
    else:
        device = torch.device("cpu")        #if nothing is found use the CPU
    print(f"Using device: {device}")

    return device


def plot_train_loss(train_losses, val_losses, experiment_name):
    """
    Plots the training and validation loss over epochs and saves the plot as an image.
    Args:
        train_losses (list or array-like): A list or array containing the training loss values for each epoch.
        val_losses (list or array-like): A list or array containing the validation loss values for each epoch.
        experiment_name (str): A string representing the name of the experiment, used to name the saved plot file.
    Saves:
        A PNG image of the plot in the directory specified by the `IMAGES_DIR` constant, 
        with the filename formatted as "loss_<experiment_name>.png".
    Displays:
        The plot of training and validation loss.
    """
    
    # Plot training/validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    filename = f"loss_{experiment_name}.png"
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(MODEL_FOLDER, filename))
    plt.show()


def plot_train_loss(train_losses, val_losses, experiment_name='train_loss'):
    """
    Plots the training and validation loss over epochs and saves the plot as an image.
    Args:
        train_losses (list or array-like): A list or array containing the training loss values for each epoch.
        val_losses (list or array-like): A list or array containing the validation loss values for each epoch.
        experiment_name (str): A string representing the name of the experiment, used to name the saved plot file.
    Saves:
        A PNG image of the plot in the directory specified by the `IMAGES_DIR` constant, 
        with the filename formatted as "loss_<experiment_name>.png".
    Displays:
        The plot of training and validation loss.
    """
    
    # Plot training/validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    filename = f"loss_{experiment_name}.png"
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(MODEL_FOLDER, filename))
    plt.show()


def train_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer, patience):
    """
    Trains and validates a PyTorch model, with support for early stopping and TensorBoard logging.
    Args:
        model (torch.nn.Module): The PyTorch model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function to calculate the error.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics and histograms.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    Returns:
        tuple: A tuple containing:
            - train_losses (list): List of training losses for each epoch.
            - val_losses (list): List of validation losses for each epoch.
            - model (torch.nn.Module): The trained model.
            - writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer used for logging.
    Notes:
        - The function logs training and validation losses to TensorBoard.
        - Early stopping is triggered if the validation loss does not improve for a specified number of epochs.
        - Histograms of model parameters are logged to TensorBoard at each epoch.
    """

    # empty list to store the training and validation losses
    train_losses, val_losses = [], []

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs): # loop through every epoch
        # Start timing the epoch
        epoch_start_time = time.time()

        # Training
        model.train() # The model should be in training mode to use batch normalization and dropout
        train_loss = 0
        for batch_no, (batch_x, batch_y) in enumerate(train_loader): # loop through every batch

            # move tensors to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # set the gradients to zero
            optimizer.zero_grad() 

            # Forward pass
            predictions = model(batch_x) # make a prediction with the current model, uses the forward() method
            loss = criterion(predictions, batch_y) # calculate the loss based on the prediction

            # Backward pass and optimization
            loss.backward() # calculated the gradiets for the given loss
            optimizer.step() # updates the weights and biases for the given gradients
            train_loss += loss.item() # calulate loss per batch

            txt_train = f"\t<train> Epoch: {epoch}, Batch: {batch_no}/{len(train_loader)} ({100*batch_no/len(train_loader):.2f}%), Loss: {train_loss:.6f}, Time: {(time.time() - epoch_start_time):.2f}s"
            print(txt_train, end='\r')  # Overwrite batch line

        train_loss /= len(train_loader) # calulate loss per epoch
        train_losses.append(train_loss)

        writer.add_scalar("train_loss", train_loss, epoch)

        # Validation
        model.eval() # The model should be in eval mode to not use batch normalization and dropout
        val_loss = 0
        with torch.no_grad(): # make sure the gradients are not changed in this step
            for batch_no, (batch_x, batch_y) in enumerate(val_loader):

                # move tensors to device
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # forward pass
                predictions = model(batch_x) # make a prediction with the current model
                loss = criterion(predictions, batch_y) # calculate the loss based on the prediction
                val_loss += loss.item() # calulate loss per batch

                txt_val = f"\t<validation> Epoch: {epoch}, Batch: {batch_no}/{len(val_loader)} ({100*batch_no/len(val_loader):.2f}%), Loss: {val_loss:.6f}, Time: {(time.time() - epoch_start_time):.2f}s"
                print(' ' * (len(txt_train)+8), end='\r')
                print(txt_val, end='\r')  # Overwrite batch line

        val_loss /= len(val_loader) # calulate loss per epoch
        val_losses.append(val_loss)

        writer.add_scalar("val_loss", val_loss, epoch)

        # Print progress (print every epoch)
        epoch_duration = time.time() - epoch_start_time
        print(' ' * (len(txt_val)+8), end='\r')
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Duration = {epoch_duration:.2f}s", end='\r')  # Overwrite epoch line
        print()  # Clear line after last batch

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"\tNo improvement in validation loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

    # Save the model
    print("Model trained, saving...")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    torch.save(model.state_dict(), TORCH_MODEL_PATH)
    print(f"Model saved to {TORCH_MODEL_PATH}")

    return train_losses, val_losses, model, writer, epoch


def test_model(model, test_loader, criterion, device):
    """
    Evaluates a trained model on a test dataset and computes the loss and predictions.
    Args:
        model (torch.nn.Module): The trained PyTorch model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used to compute the loss.
        device (torch.device): Device on which the computation will be performed (e.g., 'cpu' or 'cuda').
    Returns:
        tuple: A tuple containing:
            - test_predictions (list): List of numpy arrays containing the model's predictions for each batch.
            - true_labels (list): List of numpy arrays containing the true labels for each batch.
    Notes:
        - The model is set to evaluation mode using `model.eval()` to disable dropout and batch normalization.
        - Gradients are not computed during evaluation by using `torch.no_grad()`.
        - The input tensors are moved to the specified device, and an additional channel dimension is added for Conv1d compatibility.
        - The function prints the final test loss averaged over all batches.
    """

    test_predictions, true_labels = [], []
    model.eval() # The model should be in eval mode to not use batch normalization and dropout
    test_loss = 0
    with torch.no_grad(): # make sure the gradients are not changed in this step
        for batch_x, batch_y in test_loader:

            # move tensors to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # forward pass
            predictions = model(batch_x) # make a prediction with the current model
            loss = criterion(predictions, batch_y) # calculate the loss based on the prediction
            test_loss += loss.item() # calulate loss per batch

            # Append predictions and true labels for plotting
            test_predictions.append(predictions.cpu().numpy())
            true_labels.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader) # calculate total loss
    print(f"Final Test Loss: {test_loss:.4f}")

    return test_predictions, true_labels


def new_model(experiment_name, model, criterion, device, learning_rate, train_loader, val_loader, num_epochs, model_path, patience=10):
    """
    Trains a given model, validates it, logs training progress to TensorBoard, and saves the trained model to a specified path.
    Args:
        experiment_name (str): Name of the experiment for TensorBoard logging.
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): Loss function used for training.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        learning_rate (float): Learning rate for the optimizer.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of epochs to train the model.
        model_path (str): File path to save the trained model.
    Returns:
        tuple: A tuple containing:
            - train_losses (list): List of training losses for each epoch.
            - val_losses (list): List of validation losses for each epoch.
    """
    #Initialize the model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # define writer for tensorboard logging
    writer = SummaryWriter(f"{os.path.join(MAIN_FOLDER, 'runs')}/{experiment_name}")

    # Train the model
    train_start_time = time.time()
    train_losses, val_losses, model, writer, epoch = train_validate_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device,
        writer,
        patience,
    )
    train_duration = time.time() - train_start_time
    print(f"\nTraining completed in {train_duration:.2f} seconds")
    print(f"Training time: {train_duration/epoch:.2f} seconds/epoch\n")

    #close the tensorboard writer
    writer.flush()
    writer.close()

    # Save the model
    print("Model trained, saving...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return train_losses, val_losses


def load_model(model_file, model):
    """
    Load the weights of a pre-trained model from a specified file path.
    Args:
        model_path (str): The file path to the saved model weights.
        model (torch.nn.Module): The PyTorch model instance to load the weights into.
    Returns:
        None
    """
    # Load the model
    print("Loading existing model...")
    model.load_state_dict(torch.load(model_file, weights_only=True))
    print(f"Weights loaded to Model from {model_file}")


def create_load_model(model_file, model, **kwargs):
    """
    Create or load a machine learning model.
    This function either creates a new model and trains it or loads an existing model 
    from the specified file. If a model file does not exist, it requires additional 
    parameters to train a new model. If a model file exists, the user is prompted to 
    decide whether to train a new model or load the existing one.
    Args:
        model_file (str): Path to the model file.
        model (torch.nn.Module): The PyTorch model to be trained or loaded.
        **kwargs: Additional keyword arguments:
            - diffusion (object, optional): Diffusion process object for training.
            - lr (float, optional): Learning rate for the optimizer.
            - n_epochs (int, optional): Number of training epochs.
            - train_loader (DataLoader, optional): DataLoader for training data.
            - val_loader (DataLoader, optional): DataLoader for validation data.
            - device (torch.device, optional): Device to run the model on (e.g., 'cpu' or 'cuda').
    Raises:
        ValueError: If required arguments for creating or training a new model are missing.
    Behavior:
        - If the model file does not exist, a new model is created and trained.
        - If the model file exists, the user is prompted to either train a new model 
          or load the existing one.
        - Training losses and validation losses are plotted after training a new model.
    Note:
        Ensure all required arguments are provided when creating or training a new model.
    """
    experiment_name = kwargs.pop('experiment_name', 'IceCube') 
    criterion = kwargs.pop('criterion', None)
    device = kwargs.pop('device', None)
    learning_rate = kwargs.pop('learning_rate', None)
    train_loader = kwargs.pop('train_loader', None)
    val_loader = kwargs.pop('val_loader', None)
    num_epochs = kwargs.pop('num_epochs', None)
    patience=kwargs.pop('patience', 10)
    
    missing = [x for x in (
            experiment_name,
            criterion,
            device,
            learning_rate,
            train_loader,
            val_loader,
            num_epochs,
            patience,
        ) if x is None]

    # create new model or load existing model
    if not os.path.exists(model_file):
        print("No model found, creating a new model...")
        if missing:
            raise ValueError("Missing required arguments for creating a new model. See create_load_model()")
        train_losses, val_losses = \
            new_model(
                experiment_name, 
                model,
                criterion, 
                device, 
                learning_rate, 
                train_loader, 
                val_loader, 
                num_epochs, 
                model_file,
                patience
            )
        # plot model losses
        plot_train_loss(
            train_losses,
            val_losses,
            experiment_name,
        )

    else:
        new_model_input = input("A model exists, do you want to train a new the model? (y/n): ").strip().lower()
        print()

        if new_model_input == 'y':
            print("Preparing to train a new model...")
            if missing:
                raise ValueError("Missing required arguments for training a new model. See create_load_model()")
            train_losses, val_losses = \
                new_model(
                    experiment_name, 
                    model,
                    criterion, 
                    device, 
                    learning_rate, 
                    train_loader, 
                    val_loader, 
                    num_epochs, 
                    model_file,
                    patience
                )
            # plot model losses
            plot_train_loss(
                train_losses,
                val_losses,
                experiment_name,
            )

        # if user does not want to create a new model, load the existing model
        else:
            print("Preparing to load the model...")
            load_model(model_file, model)


def visualize_predictions(model, dataloader, device, max_images=4, threshold=0.5):
    """
    Visualize:
      1) Original image
      2) Original + ground truth mask (hard red)
      3) Original + predicted mask (hard red)
    """
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for batch in dataloader:
            # If dataloader returns (imgs, masks)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, gt_masks = batch
                gt_masks = gt_masks.to(device)
            else:
                imgs = batch
                gt_masks = None

            imgs = imgs.to(device)  # [B, H, W, 3]

            # Model prediction
            logits = model(imgs)                # [B, H, W, 1]
            probs = torch.sigmoid(logits)       # [0,1]
            pred_masks = probs.squeeze(-1)      # [B, H, W]

            for i in range(imgs.size(0)):
                if images_shown >= max_images:
                    return

                # Original image denormalized to 0-255
                orig_img = imgs[i].cpu().numpy() * 255
                orig_img = orig_img.astype(np.uint8)

                # Predicted mask
                pred_mask = pred_masks[i].cpu().numpy()
                pred_binary = pred_mask > threshold

                # Ground truth mask overlay (hard red)
                if gt_masks is not None:
                    gt_mask = gt_masks[i].squeeze(-1).cpu().numpy()
                    gt_binary = gt_mask > threshold
                    overlay_gt = orig_img.copy()
                    overlay_gt[gt_binary] = [255, 0, 0]  # paint GT pixels red
                else:
                    overlay_gt = None

                # Predicted mask overlay (hard red)
                overlay_pred = orig_img.copy()
                overlay_pred[pred_binary] = [255, 0, 0]

                # Plot 3 images
                n_cols = 3 if overlay_gt is not None else 2
                fig, axs = plt.subplots(1, n_cols, figsize=(15, 5))
                axs[0].imshow(orig_img)
                axs[0].set_title("Original")
                axs[0].axis("off")

                if overlay_gt is not None:
                    axs[1].imshow(overlay_gt)
                    axs[1].set_title("Original + GT Mask")
                    axs[1].axis("off")
                    axs[2].imshow(overlay_pred)
                    axs[2].set_title("Original + Prediction")
                    axs[2].axis("off")
                else:
                    axs[1].imshow(overlay_pred)
                    axs[1].set_title("Original + Prediction")
                    axs[1].axis("off")

                plt.show()
                images_shown += 1


def compute_pos_weight(dataloader, device='cpu'):
    """
    Computes pos_weight = num_negative / num_positive
    for BCEWithLogitsLoss
    """
    total_pos = 0
    total_neg = 0

    for batch in dataloader:
        # if dataloader returns (imgs, masks)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            masks = batch[1]
        else:
            masks = batch

        masks = masks.to(device)
        total_pos += masks.sum()
        total_neg += (1 - masks).sum()

    pos_weight = total_neg / total_pos
    return pos_weight


def dice_loss(logits, targets, smooth=1e-6):
    """
    logits: [B, H, W, 1] (raw outputs from model)
    targets: [B, H, W, 1] (binary masks, 0 or 1)
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (probs_flat * targets_flat).sum()
    dice = (2 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, dice_weight=1.0, bce_weight=1.0):
        """
        pos_weight: tensor for BCEWithLogitsLoss to balance positives
        dice_weight, bce_weight: weights for combining the two losses
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        d_loss = dice_loss(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * d_loss