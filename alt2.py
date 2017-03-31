#5
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




def runSession(dataFolder,testSplit,valSplit,batchsize,SUMMARIES_DIR,learning_rate,outModelFolder,summary,epochs = 10):
    outString = []
    outString.append("Using ALTERNATIVE22 ")
    outString.append("Using datafolder  "+str(dataFolder))
    outString.append("Using testSplit  " + str(testSplit))
    outString.append("Using valSplit  " + str(valSplit))
    outString.append("Using batchsize  "+str(batchsize))
    outString.append("Using SUMMARIES_DIR  "+str(SUMMARIES_DIR))
    outString.append("Using learning_rate  "+str(learning_rate))
    outString.append("Using outModelFolder  "+str(outModelFolder))


    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.InteractiveSession(config=config)

    # Load dataset
    import random
    seed = 100 * random.random()

    dataset = Dataset(dataFolder,batch_size=batchsize,seed=int(seed),testProp=testSplit,validation_proportion=valSplit)

    outString.append("Using dataset seed  " + str(seed))
    outString.append("Class distribution  " + str(dataset.classDistribution()))


    # Model parameters
    #input 50 batch, 96x96 images and 1 channel , shape=[None, 96,96,1]
    model_input = tf.placeholder(tf.float32, name='model_input')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    target = tf.placeholder(tf.float32, name='target')



    #------------------------------------------MODEL LAYERS
    hiddenUnits = 100
    convLayers = 3
    imsize = dataset.imageSize
    lastFilter = 128
    lastConvOut = int(imsize * (0.5 ** convLayers))

    # CONV 1
    layer_name = 'conv1'
    with tf.variable_scope(layer_name):
        conv1_out = conv_layer(model_input, [3, 3, 1, 32], layer_name)


    # CONV 2
    layer_name = 'conv2'
    with tf.variable_scope(layer_name):
        conv2_out = conv_layer(conv1_out, [3, 3, 32,32], layer_name)


    # First pooling layer
    with tf.name_scope('pool1'):
        pool1_out = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME',
                                   name='pool1')

    # CONV 3
    layer_name = 'conv3'
    with tf.variable_scope(layer_name):
        conv3_out = conv_layer(pool1_out, [3, 3, 32, 64], layer_name)


    # CONV 4
    layer_name = 'conv4'
    with tf.variable_scope(layer_name):
        conv4_out = conv_layer(conv3_out, [3,3, 64, 64], layer_name)


    # First pooling layer
    with tf.name_scope('pool2'):
        pool2_out = tf.nn.max_pool(conv4_out, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME',
                                   name='pool2')

    # CONV 3
    layer_name = 'conv5'
    with tf.variable_scope(layer_name):
        conv5_out = conv_layer(pool2_out, [3, 3, 64, 128], layer_name)

    # CONV 4
    layer_name = 'conv6'
    with tf.variable_scope(layer_name):
        conv6_out = conv_layer(conv5_out, [3, 3, 128, lastFilter], layer_name)

    # First pooling layer
    with tf.name_scope('pool3'):
        pool3_out = tf.nn.max_pool(conv6_out, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME',
                                   name='pool3')
    #HERE I ASSUME POOLING [1, 2, 2, 1] padding SAME (so it halves in every conv)

    pool3_out_flat = tf.reshape(pool3_out, [-1, lastConvOut * lastConvOut * lastFilter], name='pool3_flat')
    # Output layer  conv3 to  fc 1
    layer_name = 'fc1'
    with tf.variable_scope(layer_name):
        fc1_out = fc_layer(pool3_out_flat, [lastConvOut * lastConvOut * lastFilter, hiddenUnits], layer_name)

    fc1_out_drop = tf.nn.dropout(fc1_out, keep_prob)

    # fc2 to fc3
    layer_name = 'fc2'
    Nclasses = dataset.getNclasses()

    with tf.variable_scope(layer_name):
        fc2_out = fc_layer(fc1_out_drop, [hiddenUnits, hiddenUnits], layer_name)

    fc2_out_drop = tf.nn.dropout(fc2_out, keep_prob)

    # fc3 to output
    layer_name = 'fc3'
    Nclasses = dataset.getNclasses()

    with tf.variable_scope(layer_name):
        fc3_out = fc_layer(fc2_out_drop, [hiddenUnits, Nclasses], layer_name)


    with tf.variable_scope("fc1",reuse=True):
        rg1 = 0.01*tf.nn.l2_loss(tf.get_variable("weights"))
    with tf.variable_scope("fc2",reuse=True):
        rg2 = 0.01*tf.nn.l2_loss(tf.get_variable("weights"))
    with tf.variable_scope("fc3", reuse=True):
        rg3 = 0.01*tf.nn.l2_loss(tf.get_variable("weights"))
    regTerm = rg1 + rg2 + rg3

    #Salida con softmax + cross entropy
    with tf.name_scope('loss_function'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=fc3_out, labels=target,name='cross_entropy') + regTerm)
        if summary:
            tf.scalar_summary('cross_entropy', cross_entropy)

    # Optimization made with ADAM algorithm
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_vars = optimizer.compute_gradients(cross_entropy)
        optimizer.apply_gradients(grads_vars)
        train_step = optimizer.minimize(cross_entropy)




    # Metrics
    correct_prediction = tf.equal(tf.argmax(fc3_out, 1),
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




    outString.append("Epochs to train  " + str(epochs))
    t_i = time.time()
    n_batches = dataset.n_batches

    fallas = 0
    lasVal = 0

    trainLoss = []
    valLoss = []
    valAc = []

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
            outString.append("Epoch , training loss , accuracy "+str((epoch, loss, acc)))
            validation_accuracy, lossVal = validate(dataset,sess,accuracy,model_input,target,keep_prob,cross_entropy)

            print "Validation accuracy %f" % (validation_accuracy)
            outString.append("Validation accuracy " + str(validation_accuracy) )
            print "Validation loss %f" % (lossVal)
            outString.append("Validation lossVal " + str(lossVal) )
            outString.append("Time elapsed" + str(time.time() - t_i ) + " seconds")
            print "Time elapsed", (time.time() - t_i) / 60.0, "minutes"

            trainLoss.append(loss)
            valLoss.append(lossVal)
            valAc.append(validation_accuracy)

            if lossVal > 1:
                print "HIGH VAL LOSS ",lossVal
                batches = dataset.getValidationSet(asBatches=True)
                data = None
                labels = None
                for batch in batches:
                    data, labels = batch
                    loss = sess.run((fc2_out),
                                   feed_dict={
                                       model_input: data,
                                       target: labels,
                                       keep_prob: 1.0
                                   })
                    print "OUT ",loss
                    break


            if validation_accuracy > lasVal:
                lasVal = validation_accuracy
                fallas = 0
            else:
                fallas +=1

            if validation_accuracy == 1.0:
                print "Validation accuracy 1.0 ?!"
                # break
        if fallas == 3 :
            print "3 epochs with higher val error EARLY STOP"
            break

    #--END TRAINING test accuracy
    trainTime = time.time() - t_i

    test_acc = test(dataset,sess,accuracy,model_input,target,keep_prob)
    print "Testing set accuracy %f" % (test_acc)
    outString.append("Testing set accuracy %f" % (test_acc))

    ypred,ytrue = getPredandLabels(dataset,sess,fc3_out,model_input,keep_prob)



    saver.save(sess, outModelFolder)
    sess.close()
    tf.reset_default_graph()
    return outString ,ypred,ytrue, seed, trainTime,trainLoss, valLoss, valAc




if __name__ == "__main__":
    # ---------------------Parameters---------------------

    dataFolder = "data/CW96Scalograms"
    batchsize = 50
    SUMMARIES_DIR = 'summaries/CW96Scalograms'
    learning_rate = 1e-4
    outModelFolder = 'savedModels/CW96Scalograms'

    summary = False

    # Note the number of classes will be automatically detected from the dataset (it will check the set of image names
    # name_0, name_1 ,name_2 etc )
    l,y1,y2,seed,runTime, trainLoss, valLoss, valAc = runSession(dataFolder,0.3,0.5, batchsize, SUMMARIES_DIR, learning_rate, outModelFolder,summary,epochs=50)
    print "\n".join(l)
    # ---------------------Parameters---------------------
