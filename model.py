"""
################
### model.py ###
################

~ Will Bennett 17/06/2021

Contains the Tensorflow Transformer model 
"""
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class position_encoder: #position encoding
    def __init__(self, pos, d_model):
        self.pos = pos
        self.d_model = d_model
    
    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        return pos * angle_rates

    def positional_encoding(self):
        angle_rads = self.get_angles(np.arange(self.pos)[:, np.newaxis],
                                np.arange(self.d_model)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

#Masking

class Transformer:
    def __init__(self, ntoken, d_model, nheads, output_features, nhid=2048, nlayers=6, dropout=0.1):
        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = 

if __name__ == '__main__':
    n, d = 2048, 512
    PE = position_encoder(n, d)
    pos_encoding = PE.positional_encoding()
    print(pos_encoding.shape)
    pos_encoding = pos_encoding[0]

    # Juggle the dimensions for the plot
    pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
    pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
    pos_encoding = tf.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()
