import tensorflow as tf
from tensorflow import keras
#layers = keras.layers
from tensorflow.keras import backend
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from tensorflow.keras.applications.resnet50 import ResNet50
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
	""" 55s 1s/step
	15/50 - loss: 1.1076 - acc: 0.6375 - val_loss: 0.9047 - val_acc: 0.7053
	36/50 - loss: 0.6887 - acc: 0.7710 - val_loss: 0.5702 - val_acc: 0.8132

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
	# 15/50 - loss: 1.4403 - acc: 0.4961 - val_loss: 1.2140 - val_acc: 0.6023
	# 37/50 - loss: 1.0423 - acc: 0.6441 - val_loss: 0.8790 - val_acc: 0.7074

	if backend.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3

	# inputs: shape=(INPUT_SIZE, INPUT_SIZE, 3)
	x = inputs
	print('x:', x) # x == Tensor("input:0", shape=(?, 128, 128, 3), dtype=float32)
	#x1 = x
	x1 = layers.Lambda(lambda z: tf.image.rot90(z, k=1))(x)
	x2 = layers.Lambda(lambda z: tf.image.rot90(z, k=2))(x)
	x3 = layers.Lambda(lambda z: tf.image.rot90(z, k=3))(x)
	x4 = x
	x = layers.concatenate(
		[x1, x2, x3, x4],
		axis=channel_axis,
		name='concat0')	
	print('x1:', x1) # 
	print('concat x:', x) # Tensor("concat0/concat:0", shape=(?, 128, 128, 12), dtype=float32)

	x = conv(x, 24, 3)
	#x = conv(x, 8, 5)
	x = maxpool2(x)  # 64

	#x = bn(x)
	x = conv(x, 24, 3)
	#x = conv(x, 16, 3)
	x = maxpool2(x)  # 32

	#x = bn(x)
	x = conv(x, 24, 3)
	#x = conv(x, 16, 3)
	x = maxpool2(x)  # 16

	#x = bn(x)
	x = conv(x, 48, 3)
	#x = conv(x, 32, 3)
	x = maxpool2(x)  # 8

	x = conv(x, 48, 3)
	#x = conv(x, 32, 3)
	x = maxpool2(x)  # 4

	x = layers.Flatten()(x)
	#x = layers.Dropout(0.5)(x)
	#x = layers.Dense(200, activation='elu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(num_classes, activation='softmax', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs, x, name='cnn_128')
	
	return model

#---------

def conv_list(ls, f, k):
	return [conv(x, f, k) for x in ls]

def maxpool2_list(ls):
	return [maxpool2(x) for x in ls]
	#return list(map(maxpool2, ls))

def single_net(x, num_classes):
	x = conv(x, 24, 3)
	x = maxpool2(x)  # 64
	x = conv(x, 24, 3)
	x = maxpool2(x)  # 32
	x = conv(x, 24, 3)
	x = maxpool2(x)  # 16
	x = conv(x, 48, 3)
	x = maxpool2(x)  # 8
	x = conv(x, 48, 3)
	x = maxpool2(x)  # 4
	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(num_classes, activation='softmax')(x)
	return x


def cnn_128_rot2(inputs, num_classes):
	# 15 - loss: 1.5206 - acc: 0.4814 - val_loss: 1.4203 - val_acc: 0.5547
	# 30 - loss: 1.1992 - acc: 0.6339 - val_loss: 1.1288 - val_acc: 0.6584

	if backend.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3

	# inputs: shape=(INPUT_SIZE, INPUT_SIZE, 3)
	x0 = inputs
	print('x:', x0) # x == Tensor("input:0", shape=(?, 128, 128, 3), dtype=float32)
	#x1 = x
	x1 = layers.Lambda(lambda z: tf.image.rot90(z, k=1))(x0)
	x2 = layers.Lambda(lambda z: tf.image.rot90(z, k=2))(x0)
	x3 = layers.Lambda(lambda z: tf.image.rot90(z, k=3))(x0)
	x4 = x0

	x1 = single_net(x1, num_classes)
	x2 = single_net(x2, num_classes)
	x3 = single_net(x3, num_classes)
	x4 = single_net(x4, num_classes)

	x = tf.keras.layers.average([x1,x2,x3,x4], name=OUTPUT_NAME)

	#x = layers.Dense(num_classes, activation='softmax', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs, x, name='cnn_128')
	
	return model
