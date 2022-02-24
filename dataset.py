import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.preprocessing as preprocessing

train = preprocessing.image_dataset_from_directory(
    'Images',
    label_mode = 'categorical',
    class_names = None,
    image_size = (250, 250),
    shuffle = True,
    seed = 420,
    validation_split = 0.2,
    subset = 'training',
)

test = preprocessing.image_dataset_from_directory(
    'Images',
    label_mode = 'categorical',
    class_names = None,
    image_size = (250, 250),
    shuffle = True,
    seed = 420,
    validation_split = 0.2,
    subset = 'validation',
)

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()

        self.model.add(layers.Conv2D(8, 13, strides = 3,
            input_shape = image_size, activation = 'relu'))
        self.model.add(layers.MaxPool2D(pool_size = 2))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Conv2D(16, 3, activation = 'relu'))
        self.model.add(layers.MaxPool2D(pool_size = 2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1024, activation = 'relu'))
        self.model.add(layers.Dense(256, activation = 'relu'))
        self.model.add(layers.Dense(64, activation = 'relu'))
        self.model.add(layers.Dense(5, activation = 'softmax'))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )
    def __str__(self):
        self.model.summary()
        return ""

net = Net((250, 250, 3))
print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 200,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)