from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from data_preparation import number_of_samples

CLASSES = 43

EPOCHS = 10
BATCH_SIZE = 32
#STEPS_PER_EPOCH = number_of_samples("train") / BATCH_SIZE
#VALIDATION_STEPS = number_of_samples("valid") / BATCH_SIZE

MODEL_FILE = "model.h5"
TRAIN_DIR = "/home/andrei/work/t7_cv/trafic_signs/splited/train/"
TEST_DIR = "/home/andrei/work/t7_cv/trafic_signs/splited/valid/"
STEPS_PER_EPOCH = 27491 / BATCH_SIZE
VALIDATION_STEPS = 11806  / BATCH_SIZE


WIDTH = 299
HEIGHT = 299

# creating model
base_model = InceptionV3(include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name="avg_pool")(x)
x = Dropout(0.4)(x)
predictions = Dense(CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)


# data preparation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,  
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
)

model.save(MODEL_FILE)
