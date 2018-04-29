""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf

import numpy as np

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

keep_prob = .9
dpt = lambda x: tf.contrib.layers.dropout(x, keep_prob=keep_prob)

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    

    layer_1 = tf.layers.dense(dpt(x), n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(dpt(layer_1), n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(dpt(layer_2), num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # TODO: IMPLEMENT THIS FUNCTION
    # Define loss and optimizer
    # Compare the use of squared loss, cross entropy loss, and softmax with log-likelihood 
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

    regularization_style = 'none'

    if regularization_style == 'l1':
        reg = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l1_regularizer(scale=1.0),
            weights_list=tf.trainable_variables()[::2])
    elif regularization_style == 'l2':
        reg = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(scale=.005),
            weights_list=tf.trainable_variables()[::2])
    else:
        reg = 0

    # loss_op = tf.losses.mean_squared_error(labels=onehot_labels, predictions=pred_probas)
    # loss_op = tf.losses.log_loss(labels=onehot_labels, predictions=pred_probas)
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits) + reg

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss_op,
        global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
