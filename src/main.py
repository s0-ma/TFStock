'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("~/tensorflow/tensorflow/examples/tutorials/mnist/data/", one_hot=True)

import tfminibatch as tfmb
import tfparameter as tfp
# Import tfstock data

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

class TFNeuralNet:

    # 初期化
    def __init__(self, run_parameter, neuralnet_parameter):
        # Parameters
        self.run_parameter = run_parameter
        self.neuralnet_parameter = neuralnet_parameter

        n_classes = self.neuralnet_parameter.n_classes
        n_steps  = self.neuralnet_parameter.n_steps
        n_input = self.neuralnet_parameter.n_input
        n_hidden = self.neuralnet_parameter.n_hidden

        # tfstock data initialize
        self.tfminibatch = tfmb.TFMinibatch(n_classes)

        # tf Graph input
        self.x = tf.placeholder("float", [None, n_steps, n_input])
        self.y = tf.placeholder("float", [None, n_classes])

        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    def RNN(self, x, weights, biases):

        n_input = self.neuralnet_parameter.n_input
        n_hidden = self.neuralnet_parameter.n_hidden
        n_steps = self.neuralnet_parameter.n_steps

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    # 訓練、テスト
    def train(self):

        x = self.x
        y = self.y
        weights = self.weights
        biases = self.biases
        batch_size = self.run_parameter.batch_size
        training_iters = self.run_parameter.training_iters
        display_step = self.run_parameter.display_step
        learning_rate = self.run_parameter.learning_rate

        pred = self.RNN(x, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        softmax = tf.nn.softmax(pred)
        #cost = tf.reduce_mean(tf.square(tf.nn.softmax(pred) - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        inner_product = tf.reduce_sum(tf.mul(softmax, y), 1)
        abs_softmax = tf.sqrt(tf.reduce_sum(tf.mul(softmax, softmax), 1))
        abs_y = tf.sqrt(tf.reduce_sum(tf.mul(y, y), 1))
        abs = tf.mul(abs_softmax, abs_y)
        correct_pred_inner_product = tf.div(inner_product, abs)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_inner_product = tf.reduce_mean(tf.cast(correct_pred_inner_product, tf.float32))
        print(abs)
        print(softmax)
        print(y)
        print(accuracy)
        print(accuracy_inner_product)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = self.tfminibatch.get_next_batch_train(batch_size)
                # Reshape data to get 28 seq of 28 elements
                #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

            # Calculate accuracy for 128 mnist test images
            #test_len = 10
            test_data, test_label = self.tfminibatch.get_next_batch_test(100)
            #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
            #test_label = mnist.test.labels[:test_len]
            softmax_output = sess.run(softmax, feed_dict={x: test_data, y: test_label})

            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label}),
                  #sess.run(inner_product, feed_dict={x: test_data, y: test_label}),
                  #sess.run(accuracy, feed_dict={x: test_data, y: test_label}),
                  sess.run(accuracy_inner_product, feed_dict={x: test_data, y: test_label}),
                sess.run(softmax, feed_dict={x: test_data, y: test_label}),
                  sess.run(y, feed_dict={x: test_data, y: test_label}))
                #sess.run(tf.argmax(softmax, 1), feed_dict={x: test_data, y: test_label}))

            summary_writer = tf.train.SummaryWriter('log', graph=sess.graph)
