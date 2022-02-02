import tensorflow.keras.preprocessing as preprocessing

train = preprocessing.image_dataset_from_directory(
    'Images',
    label_mode = 'categorical',
    class_names = None,
    image_size = (250, 250),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'training',
)

test = preprocessing.image_dataset_from_directory(
    'Images',
    label_mode = 'categorical',
    class_names = None,
    image_size = (250, 250),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'validation',
)
