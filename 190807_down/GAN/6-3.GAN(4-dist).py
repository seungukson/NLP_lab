# Generative Adversarial Network (GAN) 예시
# 주어진 4개의 정규분포 데이터와 유사한 데이터를 생성한다.
#
# 2018.9.10, 아마추어퀀트 (조성현)
# -----------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

slim = tf.contrib.slim

# 데이터 세트를 생성한다.
def createDataSet(n):
    xy = []
    for i in range(n):
        r = np.random.random()
        if r < 0.25:
            x = np.random.normal(0.0, 0.5)
            y = np.random.normal(0.0, 0.5)
        elif r < 0.50:
            x = np.random.normal(0.0, 0.5)
            y = np.random.normal(2.0, 0.5)
        elif r < 0.75:
            x = np.random.normal(2.0, 0.5)
            y = np.random.normal(0.0, 0.5)
        else:
            x = np.random.normal(2.0, 0.5)
            y = np.random.normal(2.0, 0.5)
        xy.append([x, y])
    
    return pd.DataFrame(xy, columns=['x', 'y'])

# 데이터 세트를 생성한다
ds = createDataSet(n=1000)
realData = np.array(ds)
nDataRow = realData.shape[0]
nDataCol = realData.shape[1]

nGInput = 10
nGHidden = 128
nDHidden = 128

tf.reset_default_graph()
def Generator(z, nOutput=nDataCol, nHidden=nGHidden, nLayer=1):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, nOutput, activation_fn=None)
    return x

def Discriminator(x, nOutput=1, nHidden=nDHidden, nLayer=1, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h = slim.stack(x, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
        d = slim.fully_connected(h, nOutput, activation_fn=None)
    return d

def getNoise(m, n=nGInput):
    z = np.random.uniform(-1., 1., size=[m, n])
    return z

# 각 네트워크의 출력값
with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
    x = tf.placeholder(tf.float32, shape=[None, nDataCol], name='x')
    z = tf.placeholder(tf.float32, shape=[None, nGInput], name='z')
    Gz = Generator(z, nOutput=nDataCol)
    Dx = Discriminator(x)
    DGz = Discriminator(Gz, reuse=True)
    
# D-loss function. Binary cross entropy를 이용한다
# labels * -log(sigmoid(logits)) + (1 - labels) * -log(1 - sigmoid(logits))
D_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz, labels=tf.zeros_like(DGz)))
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz, labels=tf.ones_like(DGz)))

thetaG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
thetaD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

trainD = tf.train.AdamOptimizer(0.0001).minimize(D_loss, var_list = thetaD)
trainG = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list = thetaG)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

histLossD = []      # Discriminator loss history 저장용 변수
histLossG = []      # Generator loss history 저장용 변수
nBatchCnt = 5       # Mini-batch를 위해 input 데이터를 n개 블록으로 나눈다.
nBatchSize = int(realData.shape[0] / nBatchCnt)  # 블록 당 Size
nK = 1              # Discriminator 학습 횟수 (위 논문에서는 nK = 1을 사용하였음)
k = 0
for i in range(50000):
    # Mini-batch 방식으로 학습한다
    np.random.shuffle(realData)
    
    for n in range(nBatchCnt):
        # input 데이터를 Mini-batch 크기에 맞게 자른다
        nFrom = n * nBatchSize
        nTo = n * nBatchSize + nBatchSize
        
        # 마지막 루프이면 nTo는 input 데이터의 끝까지.
        if n == nBatchCnt - 1:
            nTo = realData.shape[0]
               
        # 학습 데이터를 준비한다
        bx = realData[nFrom : nTo]
        bz = getNoise(m=bx.shape[0])

        if k < nK:
            # Discriminator를 nK-번 학습한다.
            _, lossDHist = sess.run([trainD, D_loss], feed_dict={x: bx, z : bz})
            k += 1
        else:
            # Generator를 1-번 학습한다.
            _, lossGHist = sess.run([trainG, G_loss], feed_dict={x: bx, z : bz})
            k = 0
    
    # 100번 학습할 때마다 Loss, KL의 history를 보관해 둔다
    if i % 100 == 0:
        histLossD.append(lossDHist)
        histLossG.append(lossGHist)
        print("%d) D-loss = %.4f, G-loss = %.4f" % (i, lossDHist, lossGHist))
    
    # 1000번 학습할 때마다 fakeData를 그려본다
    if i % 1000 == 0:
        plt.figure(figsize=(8, 6))
        fakeData = sess.run(Gz, feed_dict={z : getNoise(m=1000)})
        plt.scatter(realData[:, 0], realData[:, 1], c='blue', s=5)
        plt.scatter(fakeData[:, 0], fakeData[:, 1], c='red', s=5)
        plt.show()

plt.figure(figsize=(8, 4))
plt.plot(histLossD, label='Loss-D')
plt.plot(histLossG, label='Loss-G')
plt.legend()
plt.title("Loss history")
plt.show()

