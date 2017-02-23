import os
import sys
import time
import numpy as np
from cnnBlocks import conv_layer, fc_layer, test, validate, getPredandLabels
import matplotlib.pyplot as plt
import tensorflow as tf
#basado en https://github.com/ignacioreyes/convnet-tutorial/blob/master/convnet-tutorial.ipynb
from dataset import Dataset
import sklearn as sk




def runSession(dataFolder,batchsize,SUMMARIES_DIR,learning_rate,outModelFolder):
    outString = ""


    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.InteractiveSession(config=config)

    # Load dataset
    import random
    seed = 100 * random.random()
    dataset = Dataset(dataFolder,batch_size=batchsize,seed=int(seed))


    """
    Architecture summary (this is the original architecture. The one in this file is a little different)

    1x32x32-64C3-64P2-64C4-
    64P2-128C3-128P2-512N-6N in CNN.

    Input image 32x32

    Conv 1 filter size 3x3 with 64 features -- pooling (max??? doesnt say) 2x2

    Conv 2 filter size 4x4 with 64 features -- pooling (max??? doesnt say) 2x2

    Conv 3 filter size 3x3 with 128 features -- pooling (max??? doesnt say) 2x2

    ---Flatten conv 3 to use as input for FC layer (fully connected)

    Fc 512 hiden units

    Output layer FC 6 units (6 units for 6 different classes)

    The FC layers have relu. The Fc layers have dropout.
    Mini batch 50 - learning rate  0.01.

    Changes in this file

    -The input is 96x96
    -Using max pooling and relu in conv layers
    -Softmax in output layer (in the paper doesnt mention or is vague)
    -Learning rate is dinamic (fixed in the paper) using adam with lr 0.0001 initial
    -Batches is a parameter but i used 5 for the test
    -The classes are 2-3 for the moment (6 in the paper)

    """

    # Model parameters
    #input 50 batch, 96x96 images and 1 channel , shape=[None, 96,96,1]
    model_input = tf.placeholder(tf.float32, name='model_input')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    target = tf.placeholder(tf.float32, name='target')



    #------------------------------------------MODEL LAYERS


    # CONV 1
    layer_name = 'conv1'
    with tf.variable_scope(layer_name):
        conv1_out = conv_layer(model_input, [3, 3, 1, 64], layer_name)
    # First pooling layer
    with tf.name_scope('pool1'):
        pool1_out = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME',
                                   name='pool1')
    # CONV 2
    layer_name = 'conv2'
    with tf.variable_scope(layer_name):
        conv2_out = conv_layer(pool1_out, [4, 4, 64, 64], layer_name)
    # Second pooling layer
    with tf.name_scope('pool2'):
        pool2_out = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME',
                                   name='pool2')

    # CONV 3
    layer_name = 'conv3'
    with tf.variable_scope(layer_name):
        conv3_out = conv_layer(pool2_out, [3, 3, 64, 128], layer_name)
    # Second pooling layer
    with tf.name_scope('pool3'):
        pool3_out = tf.nn.max_pool(conv3_out, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME',
                                   name='pool3')

    pool3_out_flat = tf.reshape(pool3_out, [-1, 12 * 12 * 128], name='pool3_flat')

    # Output layer  conv3 to  fc 1
    layer_name = 'fc1'
    with tf.variable_scope(layer_name):
        fc1_out = fc_layer(pool3_out_flat, [12 * 12 * 128, 512], layer_name)

    fc1_out_drop = tf.nn.dropout(fc1_out, keep_prob)

    # fc2 to output
    layer_name = 'fc2'
    Nclasses = dataset.getNclasses()

    with tf.variable_scope(layer_name):
        fc2_out = fc_layer(fc1_out_drop, [512, Nclasses], layer_name)

    #Salida con softmax + cross entropy
    with tf.name_scope('loss_function'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=fc2_out, labels=target,name='cross_entropy'))
        if summary:
            tf.scalar_summary('cross_entropy', cross_entropy)

    # Optimization made with ADAM algorithm
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_vars = optimizer.compute_gradients(cross_entropy)
        optimizer.apply_gradients(grads_vars)
        train_step = optimizer.minimize(cross_entropy)


    # Metrics
    correct_prediction = tf.equal(tf.argmax(fc2_out, 1),
                                  tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    if summary:
        tf.scalar_summary('accuracy', accuracy)



    # -------------------------------TRAIN ------------------------------------------------
    saver = tf.train.Saver()


    #tf.set_random_seed(1)
    if summary:
        merged = tf.merge_all_summaries()

    if summary:
        train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/train',
                                              sess.graph)
        validation_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/validation')
    sess.run(tf.initialize_all_variables())


    #--START TRAIN
    epochs = 1
    t_i = time.time()
    n_batches = dataset.n_batches

    while dataset.getEpoch() < epochs:
        epoch = dataset.getEpoch()
        batch, batch_idx = dataset.nextBatch()
        batch_data = batch[0]
        batch_labels = batch[1]
        # just a training iteration
        _ = sess.run((train_step),
                     feed_dict={
                         model_input: batch_data,
                         target: batch_labels,
                         keep_prob: 0.5
                     })
        step = batch_idx + epoch * n_batches

        if batch_idx == 0:
            loss, acc, grads = sess.run((cross_entropy, accuracy, grads_vars),
                                        feed_dict={
                                            model_input: batch_data,
                                            target: batch_labels,
                                            keep_prob: 1.0
                                        })
            print "Epoch %d, training loss %f, accuracy %f" % (epoch, loss, acc)
            validation_accuracy = validate(dataset,sess,accuracy,model_input,target,keep_prob)

            print "Validation accuracy %f" % (validation_accuracy)
            print "Time elapsed", (time.time() - t_i) / 60.0, "minutes"
            if validation_accuracy == 1.0:
                print "Validation accuracy 1.0 ?!"
                #break

    #--END TRAINING test accuracy

    test_acc = test(dataset,sess,accuracy,model_input,target,keep_prob)
    print "Testing set accuracy %f" % (test_acc)


    print "confusion_matrix"
    ypred,ytrue = getPredandLabels(dataset,sess,fc2_out,model_input,keep_prob)

    print ypred[0:10]
    print ytrue[0:10]

    print sk.metrics.confusion_matrix(ytrue, ypred)

    saver.save(sess, outModelFolder)




if __name__ == "__main__":
    # ---------------------Parameters---------------------

    dataFolder = "data/BaselineOuterInner"
    batchsize = 50
    SUMMARIES_DIR = 'summaries/BaselineOuterInner'
    learning_rate = 1e-4
    outModelFolder = 'savedModels/BaselineOuterInner'

    summary = False

    # Note the number of classes will be automatically detected from the dataset (it will check the set of image names
    # name_0, name_1 ,name_2 etc )
    runSession(dataFolder, batchsize, SUMMARIES_DIR, learning_rate, outModelFolder)
    # ---------------------Parameters---------------------