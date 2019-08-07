# Generative Adversarial Network (GAN) 예시
# tf.contrib.slim 기능을 이용하여 네트워크를 더 간결하게 표현한다.
# 참조 : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
#
# 2018.9.10, 아마추어퀀트 (조성현)
# -----------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

slim = tf.contrib.slim

# 정규분포로부터 데이터를 샘플링한다
realData = np.random.normal(size=1000)
realData = realData.reshape(realData.shape[0], 1)
nDataRow = realData.shape[0]
nDataCol = realData.shape[1]

nGInput = 8
nGHidden = 10
nDHidden = 10
nUnrollK = 5

# 데이터 P, Q에 대한 KL divergence를 계산한다.
def KL(P, Q):
    # 두 데이터의 분포를 계산한다
    histP, binsP = np.histogram(P, bins=50)
    histQ, binsQ = np.histogram(Q, bins=binsP)
    
    # 두 분포를 pdf로 만들기 위해 normalization한다.
    histP = histP / (np.sum(histP) + 1e-8)
    histQ = histQ / (np.sum(histQ) + 1e-8)

    # KL divergence를 계산한다
    kld = np.sum(histP * np.log(histP / histQ))
    return histP, histQ, kld

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

def getNoise(m, n=8):
    z = np.random.uniform(-1., 1., size=[m, n])
    return z

# 각 네트워크의 출력값
with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
    x = tf.placeholder(tf.float32, shape=[None, nDataCol], name='x')
    z = tf.placeholder(tf.float32, shape=[None, nGInput], name='z')
    Gz = Generator(z, nOutput=nDataCol)
    Dx = Discriminator(x)
    DGz = Discriminator(Gz, reuse=True)
    
# D-loss function
# Binary cross entropy를 이용한다
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
histKL = []         # KL divergence history 저장용 변수
nBatchCnt = 5       # Mini-batch를 위해 input 데이터를 n개 블록으로 나눈다.
nBatchSize = int(realData.shape[0] / nBatchCnt)  # 블록 당 Size
nK = 1              # Discriminator 학습 횟수 (위 논문에서는 nK = 1을 사용하였음)
k = 0
for i in range(10000):
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
        p, q, kld = KL(bx, sess.run(Gz, feed_dict={z : bz}))
        histKL.append(kld)
        histLossD.append(lossDHist)
        histLossG.append(lossGHist)
        print("%d) D-loss = %.4f, G-loss = %.4f, KL = %.4f" % (i, lossDHist, lossGHist, kld))

plt.figure(figsize=(6, 3))
plt.plot(histLossD, label='Loss-D')
plt.plot(histLossG, label='Loss-G')
plt.legend()
plt.title("Loss history")
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(histKL)
plt.title("KL divergence")
plt.show()

# real data 분포 (p)와 fake data 분포 (q)를 그려본다
fakeData = sess.run(Gz, feed_dict={z : getNoise(m=realData.shape[0])})
p, q, kld = KL(realData, fakeData)
plt.plot(p, color='blue', linewidth=2.0, alpha=0.7, label='Real Data')
plt.plot(q, color='red', linewidth=2.0, alpha=0.7, label='Fake Data')
plt.legend()
plt.title("Distibution of Real and Fake Data")
plt.show()
print("KL divergence = %.4f" % kld)

# Fake Data를 Discriminator에 넣었을 때 출력값을 확인해 본다.
#outputD = sess.run(Dx, feed_dict={x : fakeData})
#print(outputD.T)
#sess.close()
