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

trainX = np.random.random([100,5])
trainY = np.random.choice([0,1], 100).reshape(100,1)
testZ = np.random.random([1,5])



tf.reset_default_graph()
model = MyModel()

x = tf.placeholder(dtype=tf.float32, shape=(None, 5))
y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

output = model(x)
loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, output))
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, _loss = sess.run([train, loss], feed_dict={x: trainX, y:trainY})
    if i % 100 == 0:
        print("%d) loss = %.4f" % (i, _loss))

output = model(x)
sess.run(output, feed_dict={x: testZ})

