from image_processor import *
from helpers import *


# load model
model = tf.keras.models.load_model(MODEL_PATH)

# load images
data = Processor().load_unseen_data()

# initiate counter
i = 0

# goes through each image
for d in data:

    d = np.expand_dims(d, axis=0)

    # Predict the mask
    predicted_mask = model.predict(d)
    predicted_mask = np.squeeze(predicted_mask, axis=0)  # Shape: (x, y, 1)

    # Convert predicted probabilities to binary mask
    binary_mask = (predicted_mask > 0.5).astype(np.uint8)

    # squeeze singleton dimension
    binary_mask = binary_mask.squeeze()

    # get indexes where predict evaluates as true
    row, col = np.where(binary_mask == 1)

    # retrieve and upscale original image
    original = np.squeeze(d, axis=0)
    original = original * 255
    original = np.float32(original)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # create a copy of original image and mask it with predictions
    prediction = deepcopy(original)
    for row, col in zip(row, col):
        prediction[row][col] = [0, 0, 255]

    # path
    path_to_folder = os.path.join(PREDICTIONS_DATA_FOLDER, str(i))

    # Create the full directory path if it doesn't exist
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    # Save the predicted image and original
    cv2.imwrite(os.path.join(path_to_folder, 'prediction.png'), prediction)
    cv2.imwrite(os.path.join(path_to_folder, 'original.png'), original)

    # increment counter
    i += 1