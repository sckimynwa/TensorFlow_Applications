import tensorflow as tf;

# Tensorflow Version check
# print(tf.__version__)

# hello world
h = tf.constant("Hello") 
w = tf.constant(" World!")
hw = h+w

with tf.Session() as sess:
    ans = sess.run(hw)

print(ans)