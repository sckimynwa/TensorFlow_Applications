from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# get mnist data
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Modeling
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
