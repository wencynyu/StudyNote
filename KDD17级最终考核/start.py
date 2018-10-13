import tensorflow as tf

a = tf.constant("hello,tensorflow!")

with tf.Session() as sess:
    print(sess.run(a))
    