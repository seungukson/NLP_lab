import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.losses import mean_squared_error

def model_fn(features, labels, mode):
    nHidden = 64    # hidden layer의 neuron 개수
    nOutput = 1     # output layer의 neuron 개수
    
    xLstm = LSTM(nHidden)(features['x'])
    xOutput = Dense(nOutput)(xLstm)
    
    # 학습
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(mean_squared_error(labels, xOutput))
        optimizer = tf.train.AdamOptimizer(0.01)
        global_step = tf.train.get_global_step()
        train = optimizer.minimize(loss, global_step)
        
        return tf.estimator.EstimatorSpec(
                mode = mode,
                train_op = train,
                loss = loss)
        
    # 예측
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'result': xOutput})
