from scripts.helpers import os, cv2, np, torch, Dataset, A, \
    IMAGES_FOLDER, UNSEEN_DATA_FOLDER, PNG_RESOLUTION


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),              # Random horizontal flip
    A.VerticalFlip(p=0.5),                # Random vertical flip
    A.RandomRotate90(p=0.5),               # Rotate 0/90/180/270 deg
    A.RandomBrightnessContrast(
        brightness_limit=0.2, 
        contrast_limit=0.2, 
        p=0.5
    ),                                    # Brightness/contrast jitter
    A.Affine(
        scale=(0.9, 1.1),         # scale ±10%
        translate_percent=(0.05, 0.05),  # shift up to 5% in x and y
        rotate=(-15, 15),         # rotate between -15° and 15°
        p=0.5
    ),                                 # Slight shift, zoom, rotate
], additional_targets={'mask': 'mask'})   # Tell it masks need same transforms


class CustomDataset(Dataset):
    '''
    to handle png images
    '''

    def load_train_data(self):
        ''' 
        load images at `images` into class attributes
        '''

        self.X = []
        self.y = []

        # for each folder in `images folder`
        for folder in os.listdir(IMAGES_FOLDER):

            if folder.startswith('.'):
                continue

            # path to folder
            folder_path = os.path.join(IMAGES_FOLDER, folder)

            # for each file inside folder
            for file in os.listdir(folder_path):

                # path to file
                file_path = os.path.join(folder_path, file)

                # skip these ones
                if file == 'transformed.png':
                    continue
                
                # read image
                img = cv2.imread(file_path)

                # Resize image
                img = cv2.resize(img, (PNG_RESOLUTION[0], PNG_RESOLUTION[1]))

                # add image to correspondent list
                if file == 'untransformed.png':
                    
                    # Convert the image from BGR (OpenCV default) to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Normalize pixel values to [0, 1]
                    img = img / 255.0

                    # normalize pixel value
                    self.X.append(img)

                elif file == 'mask.png':
                    # unmask starts with 96x96x3 shape
                    # change mask to 96x96x1 shape

                    # take the mean value of axis and cast as integer
                    tmp = np.mean(img, axis=2)

                    # Add a new axis to make it 96x96x1
                    tmp = tmp[:, :, np.newaxis]

                    # normalize pixel value to 0 and 1
                    tmp = tmp / 255
                    tmp = tmp.astype(int)

                    # append changed mask
                    self.y.append(tmp)

        # convert lists to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        return (self.X, self.y)
    

    def load_unseen_data(self):
        ''' 
        load images at `unseen_data` into class attributes
        '''

        self.unseen = []

        # for each folder in `images folder`
        for folder in os.listdir(UNSEEN_DATA_FOLDER):

            # path to folder
            folder_path = os.path.join(UNSEEN_DATA_FOLDER, folder)

            # for each file inside folder
            for file in os.listdir(folder_path):

                # path to file
                file_path = os.path.join(folder_path, file)

                # skip these ones
                if file != 'untransformed.png':
                    continue
                
                # read image
                img = cv2.imread(file_path)

                # Resize image
                img = cv2.resize(img, (PNG_RESOLUTION[0], PNG_RESOLUTION[1]))

                # Convert the image from BGR (OpenCV default) to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values to [0, 1]
                img = img / 255.0

                # normalize pixel value
                self.unseen.append(img)

        # convert lists to numpy arrays
        self.unseen = np.array(self.unseen)

        return self.unseen
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]      # numpy [H, W, 3], float32 0–1
        mask = self.y[idx]      # numpy [H, W, 1], {0,1}

        # # Apply transformations
        # augmented = train_transform(image=img, mask=mask)
        # img = augmented['image']
        # mask = augmented['mask']

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
