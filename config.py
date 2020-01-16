ROOT = '/content/kitti/training'

IMAGES = 'image_2'

ANNOTATIONS = 'label_2'

NUM_CLASSES = 8

index_selects = [0, 4, 5, 6, 7]

LABEL_TO_ID = {
    'DontCare' : 0,
    'Misc': 0,
    'Car' : 1, 
    'Van' : 2, 
    'Truck' : 3, 
    'Pedestrian' : 4, 
    'Person_sitting' : 5, 
    'Cyclist' : 6, 
    'Tram' : 7
}

CUSTOM_LABELS = ['DontCare',
 'Car',
 'Van',
 'Truck',
 'Pedestrian',
 'Person_sitting',
 'Cyclist',
 'Tram',
 'Misc'
 ]