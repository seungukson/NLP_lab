import numpy as np
import tensorflow as tf
from keras.layers import Dense
x_train = np.random.random([10,8])
y_train = np.random.choice([0,1],[10,1])
x_test =np.random.random([1,8])

np.round(x_train,3)
y_train

nInput = 8
nHidden = 12
nOutput = 1

tf.reset_default_graph()

x = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

#Wh = tf.Variable(tf.truncated_normal([nInput, nHidden]),dtype=tf.float32, name ='Wh')
#Bh = tf.Variable(tf.zeros(nHidden),dtype = tf.float32, name = 'Bh')
#H1 = tf.nn.relu(tf.matmul(x,Wh)+Bh, name="H1")
H1 = Dense(nHidden,activation="relu")(x)

#Wo = tf.Variable(tf.truncated_normal([nHidden,nOutput]), dtype = tf.float32, name='Wo')
#Bo = tf.Variable(tf.zeros(nOutput), dtype = tf.float32, name = 'Bo')
#predY = tf.sigmoid(tf.matmul(H1,Wo)+Bo, name = "predY")

#wrapper: tensorflow 3line을 dense로

predY = Dense(nOutput,activation='sigmoid')(H1)

clipY = tf.clip_by_value(predY,0.000001, 0.999999)# 로그속에 0이 되는걸방지
cost = -tf.reduce_mean((1-y)*tf.log(1-predY) + y*tf.log(predY))
optimizer = tf.train.AdamOptimizer(0.05)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for i in range(1000):
#     _, cost_ = sess.run([train,cost], feed_dict={x:x_train, y:y_train})
#     if(i%100==0):
#         print(i,cost_)

# for i in range(1000):
#     for j in range(10):
#         _,cost_ = sess.run([train,cost],feed_dict={x:x_train[]})

#             o            1개
# o o o o o o o o o o o o  12개
#      o o o o o o o o   8개