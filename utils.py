import tensorflow as tf
from tensorflow.keras.activations import relu
import numpy as np


def tf_heaviside(tensor, dtype=tf.float32):
    """
    Function that returns the tensor H given by the element-wise application of the Heaviside function.
    We compute the Heaviside function H(x) = sign(relu(x) + sign(x) + 1).
    In this way, we obtain the Heaviside function: H(x) = 1 for each x>=0, H(x)=0 otherwise.
    In this way, tensorflow can compute the derivative of H using the derivatives of relu and sign functions.

    :param tensor: TF tensor
    :param dtype: output type (default tf.float32)
    :return H: element-wise Heaviside function applied to input tensor
    """
    H = tf.sign(relu(tensor) + tf.sign(tensor) + 1)
    H = tf.cast(H, dtype=dtype)

    return H


def np_heaviside(tensor):
    """
    Function that returns the tensor H given by the element-wise application of the Heaviside function.
    We compute the Heaviside function H(x) = sign(relu(x) + sign(x) + 1).
    In this way, we obtain the Heaviside function: H(x) = 1 for each x>=0, H(x)=0 otherwise.
    In this way, tensorflow can compute the derivative of H using the derivatives of relu and sign functions.

    :param tensor: numpy tensor
    :return H: element-wise Heaviside function applied to input tensor
    """
    H = np.sign(np.maximum(tensor, np.zeros_like(tensor)) + np.sign(tensor) + 1)

    return H





