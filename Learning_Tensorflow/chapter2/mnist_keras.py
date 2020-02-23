# using keras model to get datasets
import tensorflow as tf

def mnist_load():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    # Train set
    train_x = train_x.reshape([train_x.shape[0], -1])
    train_x = train_x.astype('float32') / 255
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)

    # Test set
    test_x = text_x.reshape([test_x.shape[0], -1])
    test_x = test_x.astype('float32') / 255
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)

    return (train_x, train_y), (test_x, test_y)

MINIBATCH_SIZE = 100
(train_x, train_y), (test_x, test_y) = mnist_load()
buffer_size = train_x.shape[0] + test_x.shape[0]

dataset = tf.data.Dataset.from_tensor_slices(({"image" : train_x}, train_y))
dataset = dataset.shuffle(buffer_size).repeat().batch(MINIBATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

