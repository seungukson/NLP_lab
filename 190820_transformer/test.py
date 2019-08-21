import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

myNetwork = MyModel()

# data = np.array([[1,2,3,4]])
# x = tf.placeholder(dtype = tf.float32, shape = (1,4))
# output = myNetwork(x)
# output2 = myNetwork(x)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(output, feed_dict = {x:data}))
# sess.run(tf.global_variables_initializer())
# print(sess.run(output2,feed_dict ={x:data}))

a = np.random.random([100, 5])
b = np.random.choice([0,1],100).reshape(100,1)
c = np.random.random([1,5])


# class ModelTest(tf.keras.Model):
#     def __init__(self):
#         super(ModelTest,self).__init__()
#         self.dense1 = tf.keras.layers.Dense(10,activation=tf.nn.sigmoid)
#
#
#     def call(self,inputs):
#         x = self.dense1(inputs)
#         return x
#
#     def train(self,trainX,trainY):
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer());
#             sess.run()

x = tf.placeholder(dtype = tf.float32, shape=(None, 5))
y = tf.placeholder(dtype = tf.float32, shape=(None, 1))

output = myNetwork(x)
loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, output))
optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    _, _loss = sess.run([train, loss], feed_dict={x: a, y: b})
    if i % 10 == 0:
        print("%d) loss = " % i,_loss)

# ph = tf.placeholder(dtype=tf.float32, shape = (100,5))
# myNet = ModelTest()
# output = myNet(ph)
# sess.tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(output,feed_dict)