import keras
#layers = keras.layers
from keras import backend
from keras import backend as K
from keras import layers

from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.mobilenet_v2 import MobileNetV2  #224x224.
#from keras.applications.mobilenet import MobileNet  #224x224.

OUTPUT_NAME = 'output'


def model_ResNet50(inputs):

	base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', 
		input_tensor=inputs)
	x = base_model.output
	x = layers.Dense(5, activation='sigmoid', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs=inputs, outputs=x, name='keras_ResNet50')	
	return model

# --------------------

def conv(x, f, k, s=1, p='SAME', a='relu'):
	x = layers.Conv2D(
		filters=f,
		kernel_size=(k, k),
		strides=(s, s),
		padding=p,
		activation=a, # relu, selu
		#kernel_regularizer=regularizers.l2(0.01),
		use_bias=True,
		kernel_initializer=keras.initializers.he_normal(seed=10),
		bias_initializer='zeros'
		)(x)
	return x

maxpool = lambda x, p=2, s=1: layers.MaxPool2D(pool_size=p, strides=s)(x) # ! s==1 !
maxpool2 = lambda x, p=2: layers.MaxPool2D(pool_size=p)(x)	
bn = lambda x: layers.BatchNormalization()(x)

def model_3(inputs):
	""" model_first_3 == model_first2_1
	--
	77: val_miou: 0.8053
	"""
	x = inputs
	x = conv(x, f=8, k=3, s=1, p='VALID')
	x = maxpool(x)  # 64
	
	x = bn(x)
	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x)
	
	x = bn(x)
	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x)
	x = bn(x)
	x = conv(x, f=32, k=3, s=2, p='VALID')
	x = maxpool(x)
	
	x = bn(x)
	x = conv(x, f=32, k=3, s=1, p='VALID')
	x = maxpool(x)
	
	x = bn(x)	
	x = layers.BatchNormalization()(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1000, activation='sigmoid')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid', name=OUTPUT_NAME)(x)
	#x = layers.Dense(5, activation=None)(x)
	model = keras.Model(inputs, x, name='model_first_3')
	return model



def cnn_128(inputs, num_classes):
	""" Epoch 44/500 - 456s 381ms/step 
	- loss: 1.4250e-04 - accuracy: 0.9999 - miou: 0.9370 - 
	val_loss: 0.0053 - val_accuracy: 0.9999 - val_miou: 0.7339
	"""
	x = inputs 
	x = conv(x, 8, 5)
	#x = conv(x, 8, 5)
	x = maxpool2(x)  # 64

	#x = bn(x)
	x = conv(x, 16, 3)
	#x = conv(x, 16, 3)
	x = maxpool2(x)  # 32

	#x = bn(x)
	x = conv(x, 16, 3)
	#x = conv(x, 16, 3)
	x = maxpool2(x)  # 16

	#x = bn(x)
	x = conv(x, 32, 3)
	#x = conv(x, 32, 3)
	x = maxpool2(x)  # 8

	x = conv(x, 64, 3)
	#x = conv(x, 32, 3)
	x = maxpool2(x)  # 4

	x = layers.Flatten()(x)
	#x = layers.Dropout(0.5)(x)
	#x = layers.Dense(200, activation='elu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(num_classes, activation='softmax', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs, x, name='cnn_128')
	
	return model

"""
without rotation:
Epoch 25/30 - 46s 882ms/step - loss: 0.7686 - acc: 0.7492 - val_loss: 0.6033 - val_acc: 0.8090

with rotation:
Epoch 25/30 - 49s 935ms/step - loss: 1.0445 - acc: 0.6510 - val_loss: 0.7938 - val_acc: 0.7441

"""

#---------------


def cnn_128_rot(inputs, num_classes):
	""" Epoch 44/500 - 456s 381ms/step 
	- loss: 1.4250e-04 - accuracy: 0.9999 - miou: 0.9370 - 
	val_loss: 0.0053 - val_accuracy: 0.9999 - val_miou: 0.7339
	"""
	if backend.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3

	# inputs: shape=(INPUT_SIZE, INPUT_SIZE, 3)
	x = inputs
	print('x:', x) # x == Tensor("input:0", shape=(?, 128, 128, 3), dtype=float32)
	x1 = K.image.rot90(x)
	x2 = x
	x3 = x
	x4 = x
	x = layers.concatenate(
		[x1, x2, x3, x4],
		axis=channel_axis,
		name='concat0')	
	print('concat x:', x) # Tensor("concat0/concat:0", shape=(?, 128, 128, 12), dtype=float32)

	x = conv(x, 8, 5)
	#x = conv(x, 8, 5)
	x = maxpool2(x)  # 64

	#x = bn(x)
	x = conv(x, 16, 3)
	#x = conv(x, 16, 3)
	x = maxpool2(x)  # 32

	#x = bn(x)
	x = conv(x, 16, 3)
	#x = conv(x, 16, 3)
	x = maxpool2(x)  # 16

	#x = bn(x)
	x = conv(x, 32, 3)
	#x = conv(x, 32, 3)
	x = maxpool2(x)  # 8

	x = conv(x, 64, 3)
	#x = conv(x, 32, 3)
	x = maxpool2(x)  # 4

	x = layers.Flatten()(x)
	#x = layers.Dropout(0.5)(x)
	#x = layers.Dense(200, activation='elu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(num_classes, activation='softmax', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs, x, name='cnn_128')
	
	return model