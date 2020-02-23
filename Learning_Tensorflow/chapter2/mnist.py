from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

DATA_DIR = './data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# get mnist data
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Modeling
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))

# Prediction
y_pred = tf.matmul(x, w)
y_true = tf.placeholder(tf.float32, [None, 10])

# Loss Function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
)

# Gradient step
gd_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Execute
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(NUM_STEPS):
        
        # train
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

        if(step+1) % 100 == 0:
            print(step+1, '|', sess.run(cross_entropy, feed_dict={x: batch_xs, y_true: batch_ys}))

        # test
        ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})
print("Accuracy: {:.2f}%".format(ans*100))