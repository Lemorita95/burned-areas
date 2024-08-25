from satellite_images import SatelliteImages, ee

# project credentials
project = "lnm-424214"

# initialize project
SatelliteImages.initializer(project)

# defines a region of interest (ROI) [long, lat] and date range
COORDINATES = [
    {
        "description": "Creek Fire, California, USA",
        "roi": ee.Geometry.Point([-119.5766, 37.3382]),
        "dates": ["2020-11-27", "2020-12-31"],
        'skip': 1,
    },
    {
        "description": "Black Summer Fires, East Gippsland, Victoria, Australia",
        "roi": ee.Geometry.Point([149.72, -37.45]),
        "dates": ["2019-12-01", "2020-02-29"],
        "skip": 2,
    },
    {
        "description": "Camp Fire, California, USA",
        "roi": ee.Geometry.Point([-121.4345, 39.7426]),
        "dates": ["2018-11-01", "2019-01-01"],
        "skip": 0
    },
    {
        "description": "Mendocino Complex Fire, California, USA",
        "roi": ee.Geometry.Point([-123.0985, 39.3178]),
        "dates": ["2018-07-01", "2019-01-01"],
        "skip": 0
    },
    {
        "description": "Para, Brazil",
        "roi": ee.Geometry.Point([-51.94471988999162, -6.55582424452238]),
        "dates": ["2019-08-01", "2019-09-01"],
        "skip": 0
    },
    {
        "description": "pantanal 2024",
        "roi": ee.Geometry.Point([-57.664444248082845, -18.5]),
        "dates": ["2024-06-01", "2024-07-01"],
        "skip": 0
    },
    {
        "description": "Woolsey Fire, California, USA",
        "roi": ee.Geometry.Point([-118.7715, 34.2355]),
        "dates": ["2018-11-08", "2018-12-31"]
    },
    {
        "description": "Thomas Fire, California, USA",
        "roi": ee.Geometry.Point([-119.1469, 34.5044]),
        "dates": ["2017-12-04", "2018-06-12"]
    },
    {
        "description": "Carr Fire, California, USA",
        "roi": ee.Geometry.Point([-122.6356, 40.6612]),
        "dates": ["2018-07-23", "2019-06-30"],
        "skip": 1
    },
    {
        "description": "Quebec, CA",
        "roi": ee.Geometry.Point([-76.11, 53]),
        "dates": ['2023-06-20', '2023-12-30']
    },
    {
        "description": "TC, MG",
        "roi": ee.Geometry.Rectangle([-45.314544246670664, -21.73947024890882, -45.21635394217635, -21.655892918029632]),
        "dates": ['2024-06-01', '2024-06-30']
    },
    {
        "description": "PA, MG",
        "roi": ee.Geometry.Point([-45.915085346588775, -22.248489778158422]),
        "dates": ['2024-06-15', '2024-07-05']
    },
        {
        "description": "Australian Wildfires, New South Wales, Australia",
        "roi": ee.Geometry.Point([151.2093, -33.8688]),
        "dates": ["2019-12-01", "2020-01-15"],
    },
    {
        "description": "Amazon Rainforest, Brazil",
        "roi": ee.Geometry.Point([-60.0217, -3.4653]),
        "dates": ["2020-08-01", "2020-09-10"],
    },
    {
        "description": "Mount Etna, Italy",
        "roi": ee.Geometry.Point([14.9998, 37.7344]),
        "dates": ["2020-09-05", "2020-10-20"],
    },
    {
        "description": "Great Barrier Reef, Australia",
        "roi": ee.Geometry.Point([147.6922, -18.2871]),
        "dates": ["2020-07-15", "2020-08-25"],
    },
    {
        "description": "Sahara Desert, Africa",
        "roi": ee.Geometry.Point([13.4206, 24.4734]),
        "dates": ["2020-05-10", "2020-06-15"],
    },
    {
        "description": "Iceland Volcanic Activity",
        "roi": ee.Geometry.Point([-18.0500, 63.0000]),
        "dates": ["2020-03-01", "2020-04-15"],
    },
    {
        "description": "Tokyo, Japan",
        "roi": ee.Geometry.Point([139.6917, 35.6895]),
        "dates": ["2020-11-01", "2020-12-01"],
    },
    {
        "description": "Moscow, Russia",
        "roi": ee.Geometry.Point([37.6173, 55.7558]),
        "dates": ["2020-06-10", "2020-07-10"],
    },
    {
        "description": "Dubai, UAE",
        "roi": ee.Geometry.Point([55.2708, 25.2048]),
        "dates": ["2020-04-05", "2020-05-15"],
    },
    {
        "description": "Nile River Delta, Egypt",
        "roi": ee.Geometry.Point([31.2357, 30.0444]),
        "dates": ["2020-01-20", "2020-02-15"],
    },
    {
        "description": "Paris, France",
        "roi": ee.Geometry.Point([2.3522, 48.8566]),
        "dates": ["2020-10-01", "2020-11-01"],
    },
    {
        "description": "New York City, USA",
        "roi": ee.Geometry.Point([-74.0060, 40.7128]),
        "dates": ["2020-07-01", "2020-08-01"],
    },
    {
        "description": "Buenos Aires, Argentina",
        "roi": ee.Geometry.Point([-58.3816, -34.6037]),
        "dates": ["2020-12-01", "2020-12-25"],
    },
    {
        "description": "Cape Town, South Africa",
        "roi": ee.Geometry.Point([18.4241, -33.9249]),
        "dates": ["2020-11-01", "2020-12-01"],
    },
    {
        "description": "Berlin, Germany",
        "roi": ee.Geometry.Point([13.4050, 52.5200]),
        "dates": ["2020-09-01", "2020-10-01"],
    },
    {
        "description": "Rome, Italy",
        "roi": ee.Geometry.Point([12.4964, 41.9028]),
        "dates": ["2020-08-01", "2020-09-15"],
    },
    {
        "description": "Rio de Janeiro, Brazil",
        "roi": ee.Geometry.Point([-43.1729, -22.9068]),
        "dates": ["2020-10-01", "2020-11-01"],
    },
    {
        "description": "Sydney, Australia",
        "roi": ee.Geometry.Point([151.2093, -33.8688]),
        "dates": ["2020-02-01", "2020-03-01"],
    },
    {
        "description": "Seoul, South Korea",
        "roi": ee.Geometry.Point([126.9778, 37.5665]),
        "dates": ["2020-05-01", "2020-06-01"],
    },
    {
        "description": "Toronto, Canada",
        "roi": ee.Geometry.Point([-79.3832, 43.6532]),
        "dates": ["2020-04-01", "2020-05-01"],
    },
    {
        "description": "Mumbai, India",
        "roi": ee.Geometry.Point([72.8777, 19.0760]),
        "dates": ["2020-06-01", "2020-07-01"],
    },
    {
        "description": "Lima, Peru",
        "roi": ee.Geometry.Point([-77.0369, -12.0464]),
        "dates": ["2020-05-15", "2020-06-15"],
    },
    {
        "description": "Jakarta, Indonesia",
        "roi": ee.Geometry.Point([106.8456, -6.2088]),
        "dates": ["2020-09-01", "2020-10-01"],
    },
    {
        "description": "Singapore",
        "roi": ee.Geometry.Point([103.8198, 1.3521]),
        "dates": ["2020-08-01", "2020-09-01"],
    },
    {
        "description": "Hong Kong",
        "roi": ee.Geometry.Point([114.1694, 22.3193]),
        "dates": ["2020-07-01", "2020-08-01"],
    },
    {
        "description": "Athens, Greece",
        "roi": ee.Geometry.Point([23.8103, 37.9838]),
        "dates": ["2020-04-01", "2020-05-01"],
    },
    {
        "description": "Madrid, Spain",
        "roi": ee.Geometry.Point([-3.7038, 40.4168]),
        "dates": ["2020-03-01", "2020-04-01"],
    },
    {
        "description": "Lisbon, Portugal",
        "roi": ee.Geometry.Point([-9.1399, 38.7223]),
        "dates": ["2020-01-01", "2020-02-01"],
    },
    {
        "description": "Stockholm, Sweden",
        "roi": ee.Geometry.Point([18.0686, 59.3293]),
        "dates": ["2020-12-01", "2020-12-15"],
    },
    {
        "description": "Copenhagen, Denmark",
        "roi": ee.Geometry.Point([12.5683, 55.6761]),
        "dates": ["2020-10-01", "2020-11-01"],
    },
    {
        "description": "Oslo, Norway",
        "roi": ee.Geometry.Point([10.7519, 59.9139]),
        "dates": ["2020-11-01", "2020-12-01"],
    },
    {
        "description": "Vienna, Austria",
        "roi": ee.Geometry.Point([16.3738, 48.2082]),
        "dates": ["2020-05-01", "2020-06-01"],
    },
    {
        "description": "Zurich, Switzerland",
        "roi": ee.Geometry.Point([8.5417, 47.3769]),
        "dates": ["2020-03-01", "2020-04-01"],
    },
    {
        "description": "Brussels, Belgium",
        "roi": ee.Geometry.Point([4.3499, 50.8503]),
        "dates": ["2020-06-01", "2020-07-01"],
    },
    {
        "description": "Warsaw, Poland",
        "roi": ee.Geometry.Point([21.0122, 52.2298]),
        "dates": ["2020-07-01", "2020-08-01"],
    },
    {
        "description": "Budapest, Hungary",
        "roi": ee.Geometry.Point([19.0402, 47.4979]),
        "dates": ["2020-05-15", "2020-06-15"],
    },
    {
        "description": "Prague, Czech Republic",
        "roi": ee.Geometry.Point([14.4208, 50.0880]),
        "dates": ["2020-09-01", "2020-10-01"],
    },
    {
        "description": "Dubai, UAE",
        "roi": ee.Geometry.Point([55.2708, 25.2048]),
        "dates": ["2020-06-01", "2020-07-01"],
    },
    {
        "description": "Doha, Qatar",
        "roi": ee.Geometry.Point([51.5310, 25.2760]),
        "dates": ["2020-04-01", "2020-05-01"],
    },
    {
        "description": "Kuala Lumpur, Malaysia",
        "roi": ee.Geometry.Point([101.6869, 3.139]),
        "dates": ["2020-03-01", "2020-04-01"],
    },
    {
        "description": "Riyadh, Saudi Arabia",
        "roi": ee.Geometry.Point([46.7386, 24.7136]),
        "dates": ["2020-05-01", "2020-06-01"],
    },
    {
        "description": "Sao Paulo, Brazil",
        "roi": ee.Geometry.Point([-46.6333, -23.5505]),
        "dates": ["2020-08-01", "2020-09-01"],
    },
    {
        "description": "Lagos, Nigeria",
        "roi": ee.Geometry.Point([3.3792, 6.5244]),
        "dates": ["2020-07-01", "2020-08-01"],
    },
    {
        "description": "Cairo, Egypt",
        "roi": ee.Geometry.Point([31.2357, 30.0444]),
        "dates": ["2020-05-01", "2020-06-01"],
    },
    {
        "description": "Nairobi, Kenya",
        "roi": ee.Geometry.Point([36.8219, -1.2921]),
        "dates": ["2020-08-01", "2020-09-01"],
    },
    {
        "description": "Addis Ababa, Ethiopia",
        "roi": ee.Geometry.Point([38.7619, 9.145]),
        "dates": ["2020-04-01", "2020-05-01"],
    },
    {
        "description": "Algiers, Algeria",
        "roi": ee.Geometry.Point([3.0588, 36.7538]),
        "dates": ["2020-02-01", "2020-03-01"],
    },
    {
        "description": "Casablanca, Morocco",
        "roi": ee.Geometry.Point([-7.5898, 34.0209]),
        "dates": ["2020-06-01", "2020-07-01"],
    },
    {
        "description": "Lisbon, Portugal",
        "roi": ee.Geometry.Point([-9.1399, 38.7223]),
        "dates": ["2020-05-01", "2020-06-01"],
    },
    {
        "description": "San Francisco, USA",
        "roi": ee.Geometry.Point([-122.4194, 37.7749]),
        "dates": ["2020-12-01", "2020-12-31"],
    },
    {
        "description": "Los Angeles, USA",
        "roi": ee.Geometry.Point([-118.2437, 34.0522]),
        "dates": ["2020-11-01", "2020-11-30"],
    },
    {
        "description": "Seattle, USA",
        "roi": ee.Geometry.Point([-122.3321, 47.6062]),
        "dates": ["2020-10-01", "2020-10-31"],
    },
    {
        "description": "Miami, USA",
        "roi": ee.Geometry.Point([-80.1918, 25.7617]),
        "dates": ["2020-09-01", "2020-09-30"],
    },
    {
        "description": "Houston, USA",
        "roi": ee.Geometry.Point([-95.3698, 29.7604]),
        "dates": ["2020-08-01", "2020-08-31"],
    },
    {
        "description": "Dallas, USA",
        "roi": ee.Geometry.Point([-96.7969, 32.7767]),
        "dates": ["2020-07-01", "2020-07-31"],
    },
    {
        "description": "Atlanta, USA",
        "roi": ee.Geometry.Point([-84.3879, 33.7489]),
        "dates": ["2020-06-01", "2020-06-30"],
    },
    {
        "description": "Denver, USA",
        "roi": ee.Geometry.Point([-104.9903, 39.7392]),
        "dates": ["2020-05-01", "2020-05-31"],
    },
    {
        "description": "Philadelphia, USA",
        "roi": ee.Geometry.Point([-75.1652, 39.9526]),
        "dates": ["2020-04-01", "2020-04-30"],
    },
    {
        "description": "Boston, USA",
        "roi": ee.Geometry.Point([-71.0589, 42.3601]),
        "dates": ["2020-03-01", "2020-03-31"],
    },
    {
        "description": "Washington D.C., USA",
        "roi": ee.Geometry.Point([-77.0369, 38.9072]),
        "dates": ["2020-02-01", "2020-02-29"],
    },
]
