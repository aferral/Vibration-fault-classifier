import tensorflow as tf
import numpy as np

# Model blocks
def conv_layer(input_tensor, kernel_shape, layer_name, summary=False,pad='SAME'):
    # input_tensor b01c
    # kernel_shape 01-in-out
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases", [kernel_shape[3]],
                             initializer=tf.constant_initializer(0.0))

    if summary:
        tf.histogram_summary(layer_name + "/weights", weights)
        tf.histogram_summary(layer_name + "/biases", biases)

    # Other options are to use He et. al init. for weights and 0.01
    # to init. biases.
    conv = tf.nn.conv2d(input_tensor, weights,
                        strides=[1, 1, 1, 1], padding=pad)
    return tf.nn.relu(conv + biases)


def fc_layer(input_tensor, weights_shape, layer_name, summary=False):
    # weights_shape in-out
    weights = tf.get_variable("weights", weights_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", [weights_shape[1]],
                             initializer=tf.contrib.layers.xavier_initializer())

    if summary:
        tf.histogram_summary(layer_name + "/weights", weights)
        tf.histogram_summary(layer_name + "/biases", biases)
    mult_out = tf.matmul(input_tensor, weights)
    return tf.nn.relu(mult_out + biases)

# Useful training functions
def validate(dataset,sess,accuracy,mi,t,kp,cross_entropy):
    batches = dataset.getValidationSet(asBatches=True)
    accs = []
    losses = []
    data=None
    labels = None
    for batch in batches:
        data, labels = batch
        acc = sess.run((accuracy),
                       feed_dict={
                           mi: data,
                           t: labels,
                           kp: 1.0
                       })
        loss = sess.run((cross_entropy),
                        feed_dict={
                            mi: data,
                            t: labels,
                            kp: 1.0
                        })
        accs.append(acc)
        losses.append(loss)
    mean_acc = np.array(accs).mean()
    mean_loss = np.array(losses).mean()
    return mean_acc, mean_loss


def test(dataset,sess,accuracy,mi,t,kp):
    data=None
    labels = None
    batches = dataset.getTestSet(asBatches=True)
    accs = []
    for batch in batches:
        data, labels = batch
        acc = sess.run((accuracy),
                       feed_dict={
                           mi: data,
                           t: labels,
                           kp: 1.0
                       })
        accs.append(acc)
    mean_acc = np.array(accs).mean()
    return mean_acc

def getPredandLabels(dataset,sess,fc,mi,kp):
    # metrics
    y_p = tf.argmax(fc, 1)


    data=None
    labels = None
    batches = dataset.getTestSet(asBatches=True)

    y_pred = []
    y_true = []
    for batch in batches:
        data, labels = batch
        temp = sess.run(y_p, feed_dict={mi: data, kp: 1.0})
        for elem in temp:
            y_pred.append(elem)
        for elem in labels:
            y_true.append(np.argmax(elem))
    return y_pred, y_true