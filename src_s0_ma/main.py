#! /usr/bin/python
# coding:utf-8

import sys
import glob
import os

import numpy as np
import tensorflow as tf
import pandas as pd

import datetime
import random

from tensorflow.python.ops import rnn, rnn_cell
random.seed(0)

learning_rate = 1e-2
n_input = 30     #予測に直前何日分のデータを使うか
#n_lstm_layer = 2
n_hidden = 50

#forget_bias = 0.8   #LSTMのforget_bias
#keep_prob= 0.8       #Dropout層のkeeping probability

#n_input_dim = 0       #何銘柄使って予測するか
#n_output_dim = 1      #同時に何銘柄を予測するか
#n_output = 1    #直後何日分の予測を行うか

#n_batch = 1
#n_rep = 20000

#isFirstTry = True
#overWriteModel = True

# Parameters
training_iters = 10*10000+1
n_batch = 10 
display_step = 10

# Network Parameters
#n_input = 1 # MNIST data input (img shape: 28*28)
#n_steps = 30 # timesteps
#n_hidden = 1 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

class MiniBatch():
    n_class = 10

    CSV_TRAIN_DIR_PATH = "yahoo_data_trends_train"
    CSV_TEST_DIR_PATH = "yahoo_data_trends_test"

    batch_filenames = []

    batch_x_trainset = []
    batch_y_trainset = []

    batch_x_testset = []
    batch_y_testset = []

    def __init__(self):

        y_trainset = []
        y_testset = []

        #train
        for filename in glob.glob(os.path.join(self.CSV_TRAIN_DIR_PATH, '*.csv')):
            df = pd.read_csv(filename, index_col="Date")["Adj Close"]
            df = self._normalize(df)

            tmp = []
            tmp.append(df[:n_input].values)
            self.batch_x_trainset.append(zip(*tmp))

            self.batch_filenames.append(filename)
            y_trainset.append(df[n_input:].max())

        thresholds = self._determine_thresholds(y_trainset)
        print thresholds

        for y in y_trainset:
            tmp = [0]*self.n_class
            for i, t in enumerate(thresholds):
                if (y < t):
                    tmp[i] = 1
                    break
            if (max(tmp) == 0):
                tmp[self.n_class-1] = 1
            self.batch_y_trainset.append(tmp)

        #test
        for filename in glob.glob(os.path.join(self.CSV_TEST_DIR_PATH, '*.csv')):
            df = pd.read_csv(filename, index_col="Date")["Adj Close"]
            df = self._normalize(df)

            tmp = []
            tmp.append(df[:n_input].values)
            self.batch_x_testset.append(zip(*tmp))
            y_testset.append(df[n_input:].max())

        for y in y_testset:
            tmp = [0]*self.n_class
            for i, t in enumerate(thresholds):
                if (y < t):
                    tmp[i] = 1
                    break
            if (max(tmp) == 0):
                tmp[self.n_class-1] = 1
            self.batch_y_testset.append(tmp)

    def _normalize(self, df):
        max = df[:n_input].max()
        min = df[:n_input].min()
        return (df - min)/(max-min)

    def _determine_thresholds(self, batchset):
        ret = []
        s = sorted(batchset)
        for i in range(1, self.n_class):
            ret.append(s[len(s)/self.n_class * i])
        return ret

    def get_next_batch_train(self, batch_size):
        ret_x = []
        ret_y = []
        for i in range(batch_size):
            r = random.randint(0,len(self.batch_x_trainset)-1)
            ret_x.append(self.batch_x_trainset[r])
            ret_y.append(self.batch_y_trainset[r])
        return ret_x, ret_y

    def get_next_batch_test(self, batch_size):
        ret_x = []
        ret_y = []
        for i in range(batch_size):
            r = random.randint(0,len(self.batch_x_testset))
            ret_x.append(self.batch_x_testset[r])
            ret_y.append(self.batch_y_testset[r])
        return ret_x, ret_y



#def loss(output_op, supervisor_ph):
#    with tf.name_scope("loss") as scope:
#        square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
#        loss_op = tf.sqrt(square_error)
#        tf.summary.scalar("loss", loss_op)
#        return loss_op
#
#
#def training(loss_op):
#    with tf.name_scope("training") as scope:
#        optimizer = tf.train.AdamOptimizer(learning_rate)
#        training_op = optimizer.minimize(loss_op)
#        return training_op
#
#def _print(val, predict):
#    print val, predict
#
#def print_result(output_op, fullspan=False, label=None, isPrint=True):
#    global df
#
#    result = pd.DataFrame(index=df.index, columns=["today", "test_anser", "predict_train", "predict_test"])
#    result.loc[:,"today"] = df.loc[:, TARGET+"_"]
#
#    #全データに対して
#    for epoch in range(0,len(X)):
#        if((epoch < len(trainX)-100) and not fullspan):
#            continue
#
#        rnum = range(epoch, epoch+1)
#        inputs = np.array([ X[r] for r in rnum] * n_batch)
#        ts = np.array([ Y[r] for r in rnum] * n_batch)
#
#        pred_dict = {
#            input_ph:  inputs,
#            supervisor_ph: ts,
#            keep_prob_ph:1.
#        }
#        output = sess.run([output_op], feed_dict=pred_dict)
#        if(epoch < len(trainX)):
#            result.iloc[epoch+n_input-n_output, 1:4] = [ts[0][0][0], output[0][0][0][0], np.nan]
#        else:
#            result.iloc[epoch+n_input-n_output, 1:4] = [ts[0][0][0], np.nan, output[0][0][0][0]]
#
#    result.to_csv(OUTPUT_FILE % str(label))
#    if(isPrint):
#        print result


# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, n_classes])

# Define weights
tf.set_random_seed(0)
weights = {
    'out': tf.Variable(tf.truncated_normal(
        [n_hidden, n_classes],mean=0.0, stddev=0.1))
}
biases = {
    'out': tf.Variable(tf.truncated_normal(
        [n_classes],mean=0.0, stddev=0.1))
}

def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, 1])
    x = tf.split(0, n_input, x)

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
pred_amax = tf.argmax(pred,1)
y_amax = tf.argmax(y,1)
correct_pred = tf.equal(pred_amax, y_amax)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

if True:

#with tf.Graph().as_default():
    #print "start sess"
    ##PlaceHolders
    #input_ph = tf.placeholder(tf.float32, [None, n_input, n_input_dim*2], name="input")
    #supervisor_ph = tf.placeholder(tf.float32, [None, n_output, n_output_dim], name="supervisor")
    #keep_prob_ph = tf.placeholder(tf.float32)

    ##学習パラメータ
    #weight1_var = tf.Variable(tf.truncated_normal(
    #    [n_input, n_hidden], stddev=0.1), name="weight1")
    #bias1_var = tf.Variable(tf.truncated_normal([n_hidden], stddev=0.1), name="bias1")

    #weight2_var = tf.Variable(tf.truncated_normal(
    #    [n_hidden, n_output*n_output_dim], stddev=0.), name="weight2")
    #bias2_var = tf.Variable(tf.truncated_normal([n_output*n_output_dim], stddev=0.), name="bias2")

    ##入力
    #in1 = tf.transpose(input_ph, [1, 0, 2])
    #in2 = tf.reshape(in1, [-1, n_input])
    #in3 = tf.matmul(in2, weight1_var) + bias1_var
    #in4 = tf.split(0, n_input_dim*2, in3)

    ##LSTM層
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
    #    n_hidden, forget_bias=forget_bias, state_is_tuple=False)
    ##dropout層
    #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob_ph)
    ##LSTMのmulti layer化
    #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_lstm_layer, state_is_tuple=False)

    ##学習パラメータ(lstmの内部state)
    #state_var = tf.Variable(cell.zero_state(n_batch, tf.float32), name="state")

    ##RNN
    #rnn_output, states_op = tf.nn.rnn(cell, in4, initial_state=state_var)

    ##出力の変換(hidden_nodeがそのまま出てきてしまうので)
    #out1 = tf.matmul(rnn_output[-1], weight2_var) + bias2_var
    #out2 = tf.split(1, n_output, out1)
    #output_op = tf.transpose(out2, [1, 0, 2])

    ##ロスの見積もりとoptimize
    #loss_op = loss(output_op, supervisor_ph)
    #training_op = training(loss_op)

    #summary_op = tf.summary.merge_all()
    #saver = tf.train.Saver()

    #init =tf.global_variables_initializer()

    with tf.Session() as sess:
        minibatch = MiniBatch()

        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * n_batch < training_iters:
            batch_x, batch_y = minibatch.get_next_batch_train(n_batch)
            # Reshape data to get 28 seq of 28 elements
            #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            #for i in range(len(batch_x)):
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if step % display_step == 0:

                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

                for i in range(10):
                    print(i, batch_x[i][28:30],sess.run(pred, feed_dict={x: batch_x})[i], sess.run(pred_amax, feed_dict={x: batch_x})[i], batch_y[i])

                print("Iter " + str(step*n_batch) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data, test_label = minibatch.get_next_batch_test(n_batch)
        #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        #test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
        summary_writer = tf.train.SummaryWriter('log', graph=sess.graph)

#    print "start loop"
#    with tf.Session() as sess:
#        summary_writer = tf.summary.FileWriter(log_dir_path, graph=sess.graph)
#        sess.run(init)
#
#        if not isFirstTry:
#            saver.restore(sess, model_dir_path)
#
#        minibatch = MiniBatch()
#
#        for epoch in range(int(n_rep)):
#            inputs, supervisors = minibatch.get_next_batch_train(n_batch)
#            train_dict = {
#                input_ph:      inputs,
#                supervisor_ph: supervisors,
#                keep_prob_ph:  keep_prob
#            }
#            sess.run(training_op, feed_dict=train_dict)
#
#            if((epoch+1) % 200 == 0):
#                summary_str, train_loss, final_state = sess.run([summary_op, loss_op, states_op], feed_dict=train_dict)
#                summary_writer.add_summary(summary_str, epoch)
#                print("train#%d, train loss: %e" % (epoch+1, train_loss))
#                if((epoch+1) % 2000 == 0):
#                    print_result(output_op, label=str(epoch+1), isPrint=False)
#                    saver.save(sess, model_dir_path)
#
#        print_result(output_op, fullspan=True)
#        #datas = sess.run(datas_op)
#
#        if overWriteModel:
#            saver.save(sess, model_dir_path)
