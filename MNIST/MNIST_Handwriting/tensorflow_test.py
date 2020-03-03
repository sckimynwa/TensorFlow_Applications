import tensorflow as tf

sess = tf.Session()
# constant
x = tf.constant([5], dtype=tf.float32, name='test')

# variable
y = tf.Variable([2], dtype=tf.float32, name='test2')

# placeholder
input_data = [1, 2, 3]
z = tf.placeholder(dtype=tf.float32)
z2 = z*2

sess = tf.Session()
print(sess.run(z2, feed_dict={z:input_data}))