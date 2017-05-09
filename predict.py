import numpy as np
import pylab as plt
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import rescale
import os
import time

from sklearn.cluster import KMeans
import sys

modelFile = sys.argv[1]

print "About to test random images with model ",modelFile
totalTime = 0
testCases = 100
res = []
with tf.Session() as sess:
    iamgeSize = (96,96,1)

    new_saver = tf.train.import_meta_graph(modelFile+'.meta')
    new_saver.restore(sess, modelFile)

    md = sess.graph.get_tensor_by_name('model_input:0')
    kp = sess.graph.get_tensor_by_name('dropout_prob:0')

    # Layer visualization
    outfc2 = sess.graph.get_tensor_by_name('fc2/Relu:0')


    for i in range(testCases):
        imageToUse = np.random.rand(96,96,1)
        st = time.time()
        v_ = sess.run((outfc2), feed_dict={md: imageToUse, kp: 1.0})
        res.append(time.time()-st)
    print "Time resuls mean ",np.mean(res)," std ",np.std(res)