#-*- coding: utf-8 -*-
import tensorflow as tf

from configs import DEFINES

# LSTM 단층 네트워크 구성하는 부분
def make_lstm_cell(mode, hiddenSize, index):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, name = "lstm" + str(index)) # 인코더 디코더 lstm을 한줄로 만듦
    #hiddensize : lstm 윗부분의 dimension
    if mode == tf.estimator.ModeKeys.TRAIN:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DEFINES.dropout_width)
        #dropout_width=0.9이므로 0.1이 dropout, keep을 없애기로했음.참고
    return cell
#"가끔 궁금해"=[9310, 17707, 0, 0 ...]
#9310->one_hot:0 0 0 ... 1  0 0 0 ...
# 에스티메이터 모델 부분이다.
def model(features, labels, mode, params):
    # LSTM에 입력될 워드 임베딩 레이어를 생성한다.
    # 워드 임베딩 레이어의 W는 인코더와 디코더에서 공동으로 사용한다.
    initializer = tf.contrib.layers.xavier_initializer()
    embedding = tf.get_variable(name = "embedding",
                                shape=[params['vocabulary_length'], params['embedding_size']],
                                dtype=tf.float32,
                             	initializer=initializer)
    #케라스로 안한이유. embedding을 만들어서 embedding encoder의 params와 decoder의 params 를 동일하게 해주기위해.

    # 인코더/디코더에 들어갈 워드 임베딩 결과
    # 워드 임베딩 레이어의 W는 일반 레이어와 달리, 입력층이 one-hot이므로,

    # 행렬 곱셈 (xW + b)이 필요없고 lool-up table로 처리하면 된다.
    embedding_encoder = tf.nn.embedding_lookup(params = embedding, ids = features['input']) #lookup으로 찾아라.
    embedding_decoder = tf.nn.embedding_lookup(params = embedding, ids = features['output'])

    #여기까지가 인코더 디코더 만든..10*128 사이즈 만든것..

    with tf.variable_scope('encoder_scope', reuse=tf.AUTO_REUSE):
        encoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(DEFINES.layer_size)]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
        
        # rnn_cell에 의해 지정된 dynamic_rnn 반복적인 신경망을 만든다. 
        # encoder_states 최종 상태  [batch_size, cell.state_size]
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                        inputs=embedding_encoder,
                                                        dtype=tf.float32)

    with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
        decoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(DEFINES.layer_size)]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
        decoder_outputs, decoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                        inputs=embedding_decoder,
                                                        initial_state=encoder_states,
                                                        dtype=tf.float32)

    # logits는 마지막 히든레이어를 통과한 결과값이다.
    logits = tf.layers.dense(decoder_outputs, params['vocabulary_length'], activation=None)

    ##### 예측 #####
    if mode == tf.estimator.ModeKeys.PREDICT:
        # 여기서는 softmax 출력을 그대로 내 보내고, chat.py에서 이 값을 이용하여 확률적
        # 답변을 생성한다.
        predict = tf.nn.softmax(logits)
        return tf.estimator.EstimatorSpec(mode, predictions={'indexs': predict})
    
    # logits과 같은 차원을 만들어 마지막 결과 값과 정답 값을 비교하여 에러를 구한다.
    labels_ = tf.one_hot(labels, params['vocabulary_length'])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    
    ##### 평가 #####
    if mode == tf.estimator.ModeKeys.EVAL:
        # argmax를 통해서 최대 값을 가져 온다.
        predict = tf.argmax(logits, 2)
        
        # 라벨과 결과가 일치하는지 빈도 계산을 통해 정확도를 측정하는 방법이다.
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')
    
        # accuracy를 전체 값으로 나눠 확률 값으로 한다.
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        
        # 에러 값(loss)과 정확도 값(eval_metric_ops) 전달
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    ##### 학습 #####
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  
    
        # 에러 값(loss)과 그라디언트 반환값 (train_op) 전달
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
