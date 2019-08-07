# Generative Adversarial Network (GAN) 예시
# 기본 GAN (original GAN) 알고리즘을 이용하여 정규분포에서 샘플링한 데이터 (Real Data)와
# 유사한 Fake Data를 생성한다. Real Data와 Fake Data의 분포가 잘 일치하는지 확인한다.
#
# 원 논문 : Ian J. Goodfellow, et, al., 2014, Generative Adversarial Nets.
# 위 논문의 Psedudo code (Section 4. Algorithm 1)를 위주로 하였음.
# 참고 : Original GAN은 unstable, mode collapse 등의 문제가 있으며, 최근 변형된 알고리즘이
#       많이 개발되어 있다. 여기서는 원 알고리즘을 사용하였으므로 결과가 다소 unstable 할 수 있다.
#
# 2018.9.10, 아마추어퀀트 (조성현)
# -----------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 정규분포로부터 데이터를 샘플링한다
realData = np.random.normal(size=1000)
realData = realData.reshape(realData.shape[0], 1)

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

# log 안에 0이 들어가는 것을 방지한다.
def myLog(x):
    return tf.log(x + 1e-8)

# Discriminator Network을 생성한다.
nDInput = realData.shape[1]
nDHidden = 8
nDOutput = 1

tf.reset_default_graph()
xavier = tf.contrib.layers.xavier_initializer()
x = tf.placeholder(tf.float32, shape=[None, nDInput], name='x')
D_Wh = tf.Variable(xavier([nDInput, nDHidden]), name='D_Wh')
D_Bh = tf.Variable(xavier(shape=[nDHidden]), name='D_Bh')
D_Wo = tf.Variable(xavier([nDHidden, nDOutput]), name='D_Wo')
D_Bo = tf.Variable(xavier(shape=[nDOutput]), name='D_Bo')
thetaD = [D_Wh, D_Bh, D_Wo, D_Bo]

# Generator Network을 생성한다
nGInput = 8
nGHidden = 4
nGOutput = nDInput

z = tf.placeholder(tf.float32, shape=[None, nGInput], name='z')
G_Wh = tf.Variable(xavier([nGInput, nGHidden]), name='G_Wh')
G_Bh = tf.Variable(xavier(shape=[nGHidden]), name='G_Bh')
G_Wo = tf.Variable(xavier([nGHidden, nGOutput]), name='G_Wo')
G_Bo = tf.Variable(xavier(shape=[nGOutput]), name='G_Bo')
thetaG = [G_Wh, G_Bh, G_Wo, G_Bo]

def Discriminator(x):
    D_Ho = tf.nn.relu(tf.matmul(x, D_Wh) + D_Bh)
    D_Out = tf.matmul(D_Ho, D_Wo) + D_Bo
    return tf.nn.sigmoid(D_Out)

def Generator(z):
    G_Ho = tf.nn.relu(tf.matmul(z, G_Wh) + G_Bh)
    G_Out = tf.matmul(G_Ho, G_Wo) + G_Bo
    return G_Out

def getNoise(m, n=nGInput):
    z = np.random.uniform(-1., 1., size=[m, n])
    return z

# 각 Network의 출력값
Gz = Generator(z)
Dx = Discriminator(x)
DGz = Discriminator(Gz)

# Loss function : 위 논문의 Section 3. 식 - (1)
# 이 함수를 쓰는 것보다 tensorflow에서 제공하는 tf.nn.sigmoid_cross_entropy_with_logits()을
# 사용하는 것이 더 좋다. 다음 예제부터 사용할 예정임.
D_loss = -tf.reduce_mean(myLog(Dx) + myLog(1 - DGz))
# G_loss = tf.reduce_mean(myLog(1 - DGz))
G_loss = -tf.reduce_mean(myLog(DGz)) # G_Loss는 이렇게 쓰는 것이 더 좋다

trainD = tf.train.AdamOptimizer(0.0001).minimize(D_loss, var_list = thetaD)
trainG = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list = thetaG)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

histLossD = []      # Discriminator loss history 저장용 변수
histLossG = []      # Generator loss history 저장용 변수
histKL = []         # KL divergence history 저장용 변수
nBatchCnt = 5       # Mini-batch를 위해 input 데이터를 n개 블록으로 나눈다.
nBatchSize = int(realData.shape[0] / nBatchCnt)  # 블록 당 Size
nK = 2              # Discriminator 학습 횟수 (위 논문에서는 nK = 1을 사용하였음)
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
        bz = getNoise(m=bx.shape[0], n=nGInput)

        if k < nK:
            # Discriminator를 nK-번 학습한다.
            _, lossDHist = sess.run([trainD, D_loss], feed_dict={x : bx, z : bz})
            k += 1
        else:
            # Generator를 1-번 학습한다.
            _, lossGHist = sess.run([trainG, G_loss], feed_dict={x : bx, z : bz})
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
fakeData = sess.run(Gz, feed_dict={z : getNoise(m=realData.shape[0], n=nGInput)})
p, q, kld = KL(realData, fakeData)
plt.plot(p, color='blue', linewidth=2.0, alpha=0.7, label='Real Data')
plt.plot(q, color='red', linewidth=2.0, alpha=0.7, label='Fake Data')
plt.legend()
plt.title("Distibution of Real and Fake Data")
plt.show()
print("KL divergence = %.4f" % kld)

# Fake Data를 Discriminator에 넣었을 때 출력값을 확인해 본다.
outputD = sess.run(Dx, feed_dict={x : fakeData})
print(outputD.T)
sess.close()
