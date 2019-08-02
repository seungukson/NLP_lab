import numpy as np
import tensorflow as tf
from keras.layers import Dense
x_train = np.random.random([100,8])
y_train = np.random.choice([0,1],[100,1])
x_test =np.random.random([1,8])

x_train
y_train
tf.reset_default_graph()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .repeat()\
    .batch(12)\
    .make_one_shot_iterator().\
    get_next()


nHidden = 12
nOutput = 1

x = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
H1 = Dense(nHidden, activation="relu")(x)

predY = Dense(nOutput, activation='sigmoid')(H1)

clipY = tf.clip_by_value(predY, 0.000001, 0.999999) # 로그속에 0이 되는걸방지
cost = -tf.reduce_mean((1-y)*tf.log(1-predY) + y*tf.log(predY))
optimizer = tf.train.AdamOptimizer(0.05)

train = optimizer.minimize(cost)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

sess.run(dataset)

for i in range(1000):
    bx, by = sess.run(dataset)
    _, cost_ = sess.run([train, cost], feed_dict={x: bx, y: by})
    if( i % 100 == 0 ):
        print("%d: lost = %.4f" % (i, cost_))

#             o            1개
# o o o o o o o o o o o o  12개
#      o o o o o o o o   8개