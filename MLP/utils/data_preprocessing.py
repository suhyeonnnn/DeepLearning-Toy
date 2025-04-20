# utils/data_preprocessing.py
import tensorflow as tf
import numpy as np

def to_onehotvec_label(index_labels, dim):
    num_labels = len(index_labels)
    onehotvec_labels = np.zeros((num_labels, dim))
    for i, idx in enumerate(index_labels):
        onehotvec_labels[i][idx] = 1.0
    return tf.convert_to_tensor(onehotvec_labels)
