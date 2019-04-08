# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

import settings
BATCH_SIZE = 64
INPUT_SIZE = 128

# this is the augmentation configuration we will use for training
"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
"""
train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.1,
		zoom_range=0.1,
		#rotation_range=20,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True,
		fill_mode='nearest')
		# loss: 0.8055 - acc: 0.7224 - val_loss: 0.6161 - val_acc: 0.8105

"""
train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
"""

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)

"""
train_datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
"""

def check_datagen_on_image(image_path):
	print(image_path)
	img = load_img(image_path)  # this is a PIL image
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory	
	#for batch in datagen.flow(x, batch_size=1,
	#				save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
	
	iterator = train_datagen.flow(x, batch_size=1, save_to_dir='preview', 
							save_prefix='cat', save_format='jpeg')
	for _ in range(2):
		batch = iterator.next()

#check_datagen_on_image('/mnt/lin2/datasets/natural_images/cat/cat_0008.jpg')


# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        settings.train_data_path,  # this is the target directory
        target_size=(INPUT_SIZE, INPUT_SIZE),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='categorical')  

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        settings.validation_data_path,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical') #'binary' - # since we use binary_crossentropy loss, we need binary labels

i = 0
for batch in train_generator:
	print(batch[0].shape)
	print(batch[1].shape)
	i += 1
	if i > 2: break

print(dir(train_generator))
print(train_generator.num_classes)
print(train_generator.classes)
print(len(train_generator.filenames))
print(train_generator.total_batches_seen)

num_classes = train_generator.num_classes
#-----------------

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications.resnet50 import ResNet50

def model_ResNet50(inputs, num_classes):
	base_model = ResNet50(weights='imagenet', include_top=False, 
		pooling='avg', input_tensor=inputs)
	x = base_model.output
	x = Dense(num_classes, activation='softmax')(x)
	model = Model(inputs=inputs, outputs=x, name='keras_ResNet50')	
	return model

inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='input')	
#model = model_ResNet50(inputs, num_classes)
#Epoch 35/50 - 48s 917ms/step - loss: 0.2145 - acc: 0.9285 - val_loss: 1.3748 - val_acc: 0.7603

from tf_keras_models import *
model = cnn_128(inputs, num_classes=num_classes)
#model = cnn_128_rot2(inputs, num_classes=num_classes)

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.compile(loss='binary_crossentropy', 
#			optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', 
			optimizer=keras.optimizers.Adagrad(lr=0.02),
			#optimizer='rmsprop', 
			metrics=['accuracy'])
print(model.summary())

history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // train_generator.batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // validation_generator.batch_size)
#model.save_weights('first_try.h5')  # always save your weights after training or during training

#model.fit(X_train, X_train, BATCH_SIZE=32, epochs=10, 
#	validation_data=(x_val, y_val))
#score = model.evaluate(x_test, y_test, BATCH_SIZE=32)

import matplotlib.pyplot as plt
train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'bo', label='training_acc')
plt.plot(epochs, val_acc, 'g-', label='validation_acc')
plt.show()