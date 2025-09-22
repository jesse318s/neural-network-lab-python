import keras, tensorflow, h5py
import numpy as np
from keras import ops
import keras as k
import tensorflow as tf

filename=input("\n name to save file: ")
print("")


def get_model():
	# Create a simple model.
	inputs = keras.Input(shape=(32,))
	outputs = keras.layers.Dense(1)(inputs)
	model = keras.Model(inputs, outputs)
	return model

def get_model_reshape(inShape,outShape):
	inputs = keras.Input(shape=(32,))
	classification_backbone = tf.keras.Model(keras.Input(shape=inShape),keras.Dense(shape=inShape),get_model(),keras.Dense(shape=outShape))
	return classification_backbone

model=get_model()
model.save(filename+".keras")



