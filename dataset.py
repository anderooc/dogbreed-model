from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.preprocessing as preprocessing

# train = preprocessing.image_dataset_from_directory(
#    'Images',
#    label_mode = 'categorical',
#    class_names = None,
#    color_mode= 'rgb',
#    image_size = (250, 250),
#    shuffle = True,
#    seed = 420,
#    validation_split = 0.2,
#    subset = 'training',
# )

# test = preprocessing.image_dataset_from_directory(
#     'Images',
#     label_mode = 'categorical',
#     class_names = None,
#     color_mode= 'rgb',
#     image_size = (224, 224),
#     shuffle = True,
#     seed = 420,
#     validation_split = 0.2,
#     subset = 'validation',
# )

dir = "/Users/andrew/Documents/AHCS/dogCV/Images/"  # Named folder wrong by accident
imgWidth, imgHeight = 225, 225
channels = 3
batchSize = 64
num_images = 50

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


def getImgs(dir):
    index = 0
    arraySize = imgWidth * imgHeight * channels
    images = np.ndarray(shape=(num_images, arraySize))
    labels = np.array([])

    for type in os.listdir(dir)[:50]:
        imgType = os.listdir(dir + type)
        labels = np.append(labels, type.split('-')[1])

        for image in imgType[:1]:
            file = os.path.join(dir, type + '/', image)
            imgData = mpimg.imread(file)
            resized = resize(imgData, (imgWidth, imgHeight), anti_aliasing=True)
            images[index, :] = resized.flatten()
            print(type, ':', image)
            index += 1

    return (images, labels)


def plot_images(instances, imagesPerRow, **options):
    imagesPerRow = min(len(instances), imagesPerRow)
    images = [instance.reshape(imgWidth, imgHeight, channels) for instance in instances]
    rows = (len(instances) - 1) // imagesPerRow + 1
    rowImgs = []
    empty = rows * imagesPerRow - len(instances)
    images.append(np.zeros((imgWidth, imgHeight * empty)))
    for row in range(rows):
        if (row == len(instances) / imagesPerRow):
            break
        rimages = images[row * imagesPerRow: (row + 1) * imagesPerRow]
        rowImgs.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(rowImgs, axis=0)
    plt.figure(figsize=(20, 20))
    plt.imshow(image, **options)
    plt.axis("off")
    plt.savefig('dogs_images.png', transparent=True, bbox_inches='tight', dpi=900)
    plt.show()


# images, labels = getImgs(dir)
# plot_images(images, 10)

class Model():
    def __init__(self, image_size):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(8, 13, strides=3, input_shape=image_size, activation='relu'))
        self.model.add(layers.MaxPool2D(pool_size=2))
        self.model.add(layers.Conv2D(16, 3, activation='relu'))
        self.model.add(layers.MaxPool2D(pool_size=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1024, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(5, activation='softmax'))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(lr=0.001)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

    def __str__(self):
        self.model.summary()
        return ''


model = Model((imgWidth, imgHeight, 3))

trainDatagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.3,
)

testDatagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.3,
)

trainGen = trainDatagen.flow_from_directory(
    dir,
    target_size=(imgWidth, imgHeight),
    color_mode='rgb',
    batch_size=batchSize,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=420
)

testGen = testDatagen.flow_from_directory(
    dir,
    target_size=(imgWidth, imgHeight),
    color_mode='rgb',
    batch_size=batchSize,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=420
)

model.fit_generator(
    trainGen,
    steps_per_epoch=nb_train_samples // batchSize,
    epochs=epochs,
    testData=testGen,
    validation_steps=nb_validation_samples // batchSize
)
