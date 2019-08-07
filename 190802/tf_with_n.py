import tensorflow as tf

nInput = 10
nHidden = 15
nOutput = 1

x = tf.placeholder(dtype="tf.float32",shape=[None, nInput])