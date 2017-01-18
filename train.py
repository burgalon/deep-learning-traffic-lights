import os

from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Convolution2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.optimizers import Adam

NUM_CHANNELS = 3
IMAGE_WIDTH = 224 # Original: 455
IMAGE_HEIGHT = 224 # Original: 256
NUM_CLASSES = 3
base_script_name = os.path.splitext(__file__)[0]
filepath=base_script_name + "-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5"

def get_generator(directory, train):
  if train:
    datagen = ImageDataGenerator(
      rescale=1./255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)
  else:
    datagen = ImageDataGenerator(rescale=1./255)
  return datagen.flow_from_directory(
    directory=directory,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=8,
    class_mode='categorical')

model = Sequential([
  Convolution2D(16, 3, 3, border_mode='same', subsample=(2, 2), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES), activation='relu'),
  MaxPooling2D(pool_size=(3, 3)),
  Dropout(0.2),

  Convolution2D(32, 3, 3, border_mode='same', activation='relu'),
  MaxPooling2D(pool_size=(3, 3)),
  Dropout(0.2),

  Convolution2D(64, 3, 3, border_mode='same', activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Dropout(0.2),

  Flatten(),
  Dense(128, activation='tanh'),
  Dropout(0.3),
  Dense(NUM_CLASSES, activation='softmax'),
])
model.summary()

directory = './data/'
train_generator = get_generator(directory+'./train', True)
validation_generator = get_generator(directory+'valid', False)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=Adam(lr=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='auto')
tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboar', histogram_freq=0, write_graph=True, write_images=True)
callbacks = [checkpoint, tensorboard]

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
  train_generator,
  samples_per_epoch=train_generator.N,
  nb_epoch=200,
  validation_data=validation_generator,
  nb_val_samples=validation_generator.N,
  callbacks=callbacks,
)

model.evaluate_generator(validation_generator, val_samples=len(validation_generator.filenames))
