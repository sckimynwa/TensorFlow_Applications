# download MNIST Datasets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# import tensorflow library
import tensorflow as tf

# set variables
x = tf.placeholder(tf.float32, [None 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros[10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])

