import urllib.request
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib

url = "https://raw.githubusercontent.com/kimdonghwi94/crawling/master/kr_daily.csv"
savename = 'kr_dailyl.csv'
urllib.request.urlretrieve(url,savename)
pand = pd.read_csv(savename)
covid = np.array(pand)

covid=covid[:,:2]
seq_length=3
dataX=[]
dataY=[]
copy=np.copy(covid)
for i in range(len(covid)):
    a = covid[i, 1]
    if i == 0:
        b = copy[0, 1]
        covid[i,1]=1
    else:
        b = copy[i - 1, 1]
        covid[i,1]=a-b
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        # print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

data_dim = 2
hidden_dim = 100
output_dim = 1
learning_rate = 0.01
iterations = 500
print(copy.shape)
covid=copy[:140,:]
test_set=copy[140:,:]
trainX, trainY = build_dataset(covid, seq_length)
testX, testY = build_dataset(test_set, seq_length)
print(trainX.shape,trainY.shape,testX.shape,testY.shape)
# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

import matplotlib.pyplot as plt
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
        targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()
