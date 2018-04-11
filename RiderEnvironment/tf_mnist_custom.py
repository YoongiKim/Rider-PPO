"""
Copyright (c) <2018> YoongiKim

 See the file license.txt for copying permission.
"""

import tensorflow as tf
import random
import os
import numpy as np
# import matplotlib.pyplot as plt
import cv2

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

# This path is for ppo.py, this path won't work in the test.py
CHECK_POINT_DIR = './RiderEnvironment/mnist_model_custom'
TRAIN_DATA_DIR = './RiderEnvironment/score_train_data'

def allfiles(path):
    names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            names.append(file.split('.')[0])

    return names

def to_binary(img):
    retval, threshold = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return np.array(threshold)

if __name__ == "__main__":
    x = []
    y = []

    for i in range(10):
        path = "{}/{}".format(TRAIN_DATA_DIR, i)
        names = allfiles(path)
        for name in names:
            img = cv2.imread("{}/{}/{}.png".format(TRAIN_DATA_DIR, i, name), cv2.IMREAD_GRAYSCALE)
            img = to_binary(img)
            x.append(img.reshape(784))
            y_one_hot = np.zeros(10)
            y_one_hot[i] = 1
            y.append(y_one_hot)

    print(np.shape(x))
    print(np.shape(y))

# hyper parameters
learning_rate = 0.0001
training_epochs = 1001
batch_size = 100

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128 * 4 * 4])
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

last_epoch = tf.Variable(0, name='last_epoch')

# initialize
#config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

start_from = sess.run(last_epoch)

if __name__ == "__main__":
    # train my model
    print('Start learning from:', start_from)

    for epoch in range(training_epochs):
        c, _, = sess.run([cost, optimizer], feed_dict={X: x, Y: y, keep_prob: 0.7})

        if(epoch % 100 == 0):
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))

        if(epoch % 1000 == 0):
            print("Saving network...")
            sess.run(last_epoch.assign(epoch + 1))
            if not os.path.exists(CHECK_POINT_DIR):
                os.makedirs(CHECK_POINT_DIR)
            saver.save(sess, CHECK_POINT_DIR + "/model", global_step=i)

    print('Learning Finished!')

def mnist_predict(image):
    x = image.reshape(1, 784)
    return sess.run(tf.argmax(hypothesis, 1), feed_dict={X: x, keep_prob: 1})[0]