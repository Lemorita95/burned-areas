from helpers import *

class SatelliteImages():
    ''''
    class to handle the acquisition of satellite images
    '''

    @classmethod
    def initializer(cls, project):
        """
        to be able to use ee.point / ee.rectangle at coordinates declaration
        initializer will be called at main() before __init__
        """
        # initialize earth engine
        ee.Initialize(project = project)


    def save_as_png(self, image, directory, folder, filename, size=f"{PNG_RESOLUTION[0]}x{PNG_RESOLUTION[1]}"):
        """"
        takes an ee.image object and saves as png
        args: 
            directory -> image collection folder
            folder -> folder to hold image to be saved
        """
        # generate image thumb url
        dimensions = size
        url = image.getThumbURL({'dimensions': dimensions, 'format': 'png'})

        # download image
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # save as png
        folder_path = os.path.join(directory, folder)
        save_image(img, folder_path, filename)

        print(f"save as '{folder}/{filename}'")


    def blend(self, image, mask):
        """
        blend masked image with red background
        """

        # apply mask to image and select desired bands
        image = image.updateMask(mask).select(['B4', 'B3', 'B2'])

        # fill masked pixels with red color
        red_color = ee.Image.constant([4000, 0, 0])

        # Apply the mask to the red image
        red_masked = red_color.updateMask(mask.Not())

        # Blend the original masked image with the red image
        blended_image = image.blend(red_masked)

        return blended_image


    def binary_mask(self, image):
        """
        apply mask based on image bands
        returns binary mask
        """

        # water index < 0
        ndwi = image.normalizedDifference(['B3', 'B8'])
        # vegetation index < 0.2
        ndvi = image.normalizedDifference(['B8', 'B4'])
        # near infrared
        nir = image.select(['B8'])
        # normalized burn ratio
        nbr = image.normalizedDifference(['B8', 'B12'])

        # combine masks
        mask = ndwi.gt(0.1).Or(nir.gt(1e3)).Or(nbr.gt(0)).Or(ndvi.lt(0.1))
        return mask


    def collect_transform(self, roi, dates, skip, filenumber, directory):
        """
        get data from satellite image and transforms to get burned area
        args: 
            region of interest,
            start and end date, 
            skip how many images,
            image folder name and 
            image directory path
        """

        # collect image
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(dates[0], dates[1]) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \

        # handle no image
        if collection.size().getInfo() == 0:
            return

        # single image
        image = ee.Image(collection.toList(collection.size()).get(skip))
        
        # clip image if its a polygon
        try:
            if roi._type == 'Polygon':
                image = image.clip(roi)
        # handle ee.geometry.point.buffer method
        except AttributeError:
            pass

        # get binary mask
        mask = self.binary_mask(image)

        # save untransformed rgb image
        self.save_as_png(image.visualize(**VIS_PARAMETERS), directory, filenumber, "untransformed.png")

        # save mask footprint
        self.save_as_png(mask.Not().visualize(), directory, filenumber, "mask.png")

        # save blend image with mask as a red background
        self.save_as_png(self.blend(image, mask).visualize(**VIS_PARAMETERS), directory, filenumber, "transformed.png")

    
    def load_all(self, coordinates, directory):
        """ 
        iterative method to collect_transform a coordinates collection
        """

        # for folder numeration
        i = 0

        for coordinate in coordinates:

            # get coordinate informations
            roi = coordinate['roi']
            dates = coordinate['dates']
            skip = coordinate['skip']

            # run collect_transform for coordinate
            self.collect_transform(roi, dates, skip, str(i), directory)

            # add value to counter
            i += 1


    def get_coordinates(self, coordinates):
        ''' 
        get predefined coordinates from coordinates.py and load into attribute
        '''
        # regions, dates for acquisition as a list
        self.coordinates = []
        for coordinate in coordinates:

            # check if skip was given
            if 'skip' in coordinate:
                # append record
                self.coordinates.append(coordinate)
            else:
                # add skip record
                coordinate['skip'] = 0
                
                # then append record
                self.coordinates.append(coordinate)
    

    def create_train_data(self, coordinates):
        '''
        acquire and save train data from coordinates.py
        '''
        self.get_coordinates(coordinates)
        self.load_all(self.coordinates, IMAGES_FOLDER)


    def create_unseen_data(self, amount):
        ''' 
        create random coordinates and get images
        '''
        # empty sample list
        samples = []

        # recent images
        dates = ['2024-01-01', '2024-06-01']

        for _ in range(amount):
            
            # generate pseudorandom coordinates
            rand_lat = round(random.uniform(-90, 90), 4)
            rand_long = round(random.uniform(-180, 180), 4)

            # add sample
            samples.append({
                'roi': ee.Geometry.Point([rand_long, rand_lat]),
                'dates': dates,
                'skip': 0,
            })

        self.load_all(samples, UNSEEN_DATA_FOLDER)