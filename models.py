from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np


class MLP(Model):
    """
    Multi layer perceptron implementation
    """
    def __init__(self, nodes_in_layers: list, activations: list, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.layer_lst = [Dense(nodes_in_layers[i], activation=activations[i],
                                kernel_initializer=tf.keras.initializers.HeNormal())
                          for i in range(len(nodes_in_layers))]
        self.out = Dense(num_outputs, kernel_initializer=tf.keras.initializers.HeNormal())

    def call(self, x, **kwargs):
        for l in self.layer_lst:
            x = l(x)
        x = self.out(x)
        if self.num_outputs > 1:  # normalize outputs such that they sum to 1
            return Softmax()(x)
        return tf.keras.activations.sigmoid(x)


class MLP_COMMITTEE():
    """
    This class is an implementation of a committee vote.
    This class is initiated with several MLP models (odd number)
    and takes a majority vote for classifying a 9-mer sequence
    """

    def __init__(self, mlp_lst):
        self.mlps = mlp_lst

    def __call__(self, x):
        """
        committee vote, returns committee decision and average probabilities for pos/neg predictions
        """
        avg_pred = np.zeros(x.shape[0])
        for mlp in self.mlps:
            # pred = tf.reshape(mlp(x), (x.shape[0], 2))
            # avg_pred += pred.numpy()
            pred = mlp(x).numpy()
            avg_pred += pred[:, 0]
        avg_pred = avg_pred / len(self.mlps)

        # using argmin as index 0 is positive for corona hence as argmin = 1 - argmax, will output 1
        # matching k-mer that are positive for corona
        return (avg_pred > 0.5).astype(int), avg_pred
