from helpers import *

class Processor():
    '''
    to handle png images
    '''

    def load_train_data(self):
        ''' 
        load images at `images` into class attributes
        '''

        self.evidence = []
        self.label = []

        # for each folder in `images folder`
        for folder in os.listdir(IMAGES_FOLDER):

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
                    self.evidence.append(img / 255.0)

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
                    self.label.append(tmp)

        return (np.array(self.evidence), np.array(self.label))        
    

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

        return np.array(self.unseen)