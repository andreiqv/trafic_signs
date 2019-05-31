import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input as prep_incV3
from keras.applications.resnet50 import ResNet50, preprocess_input as prep_res50
from keras.applications.resnet import ResNet101, ResNet152
from keras.callbacks import CSVLogger
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
#from data_preparation import number_of_samples

CLASSES = 43

POOLING_AVERAGE = "avg_pool"
ACTIVATION = "softmax"

CLASS_MODE = "categorical"
FILL_MODE = "nearest"

#OPTIMIZER = "rmsprop"
OPTIMIZER = keras.optimizers.Adagrad(lr=0.01)
OBJECTIVE_FUNCTION = "categorical_crossentropy"
LOSS_METRICS = ["accuracy"]

EPOCHS = 30
BATCH_SIZE = 64

MODEL_FILE = "model.h5"
TRAIN_DIR = "/home/andrei/work/t7_cv/trafic_signs/splited/train/"
TEST_DIR = "/home/andrei/work/t7_cv/trafic_signs/splited/valid/"
STEPS_PER_EPOCH = 27491 / BATCH_SIZE
VALIDATION_STEPS = 11806  / BATCH_SIZE

MODEL_FILE = "model.h5"
HISTORY_FILE = "history.csv"
#TRAIN_DIR = "data/train"
#TEST_DIR = "data/validation"

resnet_image_shape = (224, 224)
inception_image_shape = (299, 299)

def train(model):

    if model == "resnet":
        #base_model = ResNet50(include_top=False)
        base_model = ResNet101(include_top=False)
        preprocess_input = prep_res50
        shape = resnet_image_shape
    elif model == "inception":
        base_model = InceptionV3(include_top=False)
        preprocess_input = prep_incV3
        shape = inception_image_shape

    x = base_model.output
    x = GlobalAveragePooling2D(name=POOLING_AVERAGE)(x)
    x = Dropout(0.4)(x)
    predictions = Dense(CLASSES, activation=ACTIVATION)(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    #for layer in base_model.layers:
    #    layer.trainable = False

    model.compile(optimizer=OPTIMIZER, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

    # data preparation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode=FILL_MODE,
    )
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )    

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=shape,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
    )

    validation_generator = validation_datagen.flow_from_directory(
        TEST_DIR,
        target_size=shape,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
    )

    csv_logger = CSVLogger(HISTORY_FILE)

    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS,
        callbacks=[csv_logger],
    )

    model.save(MODEL_FILE)

    return history


#history = train("inception")
history = train("resnet")

#----------------

import matplotlib.pyplot as plt
train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'bo', label='training_acc')
plt.plot(epochs, val_acc, 'g-', label='validation_acc')
plt.savefig('_out.png')

#data = {'epochs':list(epochs),
#       'train_acc': train_acc, 'val_acc':val_acc,
#       'train_loss':train_loss, 'val_loss':val_loss}
import pandas as pd
df = pd.DataFrame(history.history)
epochs_list = list(range(1, len(history.history['acc']) + 1))
df.insert(0, 'epoch', epochs_list)
df.to_csv('_out_results.csv', float_format='%.4f', index=False)