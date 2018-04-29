# from Geron, 14_recurrent_neural_networks

# demonstrate LSTM

import numpy as np
import tensorflow as tf

n_steps = 28
n_inputs = 28

n_outputs = 10


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()



learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

n_neurons = 150
n_layers = 3
lstm_cells = None
# TO DO: Define the LSTM layers
pass
# END

multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)


loss = None
optimizer = None
# TO DO: Define the loss functrion and the optimizer (Adam)
pass
# End

training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)

accuracy = None
# TO DO: Compute the accuracy
pass
# END

init = tf.global_variables_initializer()

states

top_layer_h_state

n_epochs = 10
batch_size = 150


# TO DO: Write your session to run the program
pass
# END


        
