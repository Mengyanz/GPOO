import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MlpA(object):
    def __init__(self, num_inputs, num_outputs, num_layers=2, width=128):
        self.num_inputs     = num_inputs
        self.num_outputs    = num_outputs
        
        initialiser = tf.keras.initializers.HeNormal()
        model = tf.keras.models.Sequential()
        num_nodes = width
        activation = 'relu'
        for l in range(num_layers):
            if l == 0:
                num_input_nodes = num_inputs
            else:
                num_input_nodes = width
            if l == (num_layers - 1):
                num_nodes = num_outputs
                activation = 'softmax'
            model.add(layers.Dense(num_nodes, input_dim = num_input_nodes, 
                activation=activation, kernel_initializer=initialiser))

        self.model = model

    def __call__(self, x):
        """
        Args:
            x (nparray(float)): n x self.num_inputs representing all of the
                data to be clustered.
        """
        outputs = self.model(x)

        return tf.transpose(outputs)

    def get_weights(self):
        return self.model.weights
