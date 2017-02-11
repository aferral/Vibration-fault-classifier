import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

def getActivations(sess,layer,stimuli,x,keep_prob):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,9216],order='F'),keep_prob:1.0})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
#tf.trainable_variables()
def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-model.meta')
    new_saver.restore(sess, 'my-model')
    keep_prob = tf.placeholder("float")

    imageToUse=cv2.cvtColor(cv2.imread("data/mix/1_0.jpg"), cv2.COLOR_BGR2GRAY)
    plt.imshow(imageToUse, interpolation="nearest", cmap="gray")

    model_input = tf.placeholder(tf.float32, name='model_input')
    v_ = sess.run('target',feed_dict={model_input: np.reshape(imageToUse,[1,96,96,1]),keep_prob:1.0})
    print(v_)


    # with tf.variable_scope('conv1') as scope:
    #     getActivations(sess,scope,imageToUse,x,keep_prob)

    # Visualize conv1 features
    vc1 = False
    if vc1:
        saved_dict = {}
        for x in tf.trainable_variables():
            saved_dict[x.name] = x

        toSee = 'conv1'

        weights = saved_dict[toSee+"/weights:0"]
        print weights.eval().shape
        grid = put_kernels_on_grid(weights).eval()
        print grid.shape
        plt.imshow(grid.reshape(grid.shape[1],grid.shape[2]),interpolation='none')
        plt.show()