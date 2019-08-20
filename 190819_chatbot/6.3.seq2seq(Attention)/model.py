#-*- coding: utf-8 -*-
import tensorflow as tf

from configs import DEFINES

# 에스티메이터 모델 부분이다.
def model(features, labels, mode, params):
    # LSTM에 입력될 워드 임베딩 레이어를 생성한다.
    # 워드 임베딩 레이어의 W는 인코더와 디코더에서 공동으로 사용한다.
    initializer = tf.contrib.layers.xavier_initializer()
    embedding = tf.get_variable(name = "embedding",
                                shape=[params['vocabulary_length'], params['embedding_size']],
                                dtype=tf.float32,
                             	initializer=initializer)

    # 인코더/디코더에 들어갈 워드 임베딩 결과
    # 워드 임베딩 레이어의 W는 일반 레이어와 달리, 입력층이 one-hot이므로,
    # 행렬 곱셈 (xW + b)이 필요없고 lool-up table로 처리하면 된다.
    embedding_encoder = tf.nn.embedding_lookup(params = embedding, ids = features['input'])
    
    ##### Encoder #####
    with tf.variable_scope('encoder_scope', reuse=tf.AUTO_REUSE):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(params['hidden_size'])
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                        inputs=embedding_encoder,
                                                        dtype=tf.float32)

    ##### Decoder #####
    with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(params['hidden_size'])
        decoder_state = encoder_states
        
        # 매 타임 스텝에 나오는 아웃풋을 저장하는 리스트 두개를 만든다. 
        predict_tokens = list()     # predict_tokens 저장
        temp_logits = list()        # logits 저장

        # 디코더의 처음부터 끝까지 돌면서 매 스텝마다 어텐션을 적용한다.
        # output_token = <START> = 1
        output_token = tf.ones(shape=(tf.shape(encoder_outputs)[0],), dtype=tf.int32)#decoder의 입력 부분
        for i in range(DEFINES.max_sequence_length):   #(0-9)
            # TRAIN 모드에서 두 번째 스텝 이후에는 teacher forcing을 적용한다.
            if mode == tf.estimator.ModeKeys.TRAIN:
                if i > 0:
                    # teacher forcing
                    input_token_emb = tf.nn.embedding_lookup(embedding, labels[:, i-1])
                else:
                    input_token_emb = tf.nn.embedding_lookup(embedding, output_token)
            else:
                input_token_emb = tf.nn.embedding_lookup(embedding, output_token)

            ##### 어텐션 적용 (Additive attention) #####
            W1 = tf.keras.layers.Dense(params['hidden_size'])#128개
            W2 = tf.keras.layers.Dense(params['hidden_size'])#128개
            V = tf.keras.layers.Dense(1)
            
            # c와 h를 concat으로 합친다. <-첫 스탭은 인코더의 출력2개를 합친다.
            concat_decoder_state = tf.concat([decoder_state[0], decoder_state[1]], 1)
            
            hidden_with_time_axis = W2(concat_decoder_state)                        # (?, 256) -> (?, 128)
            hidden_with_time_axis = tf.expand_dims(hidden_with_time_axis, axis=1)   # (?, 128) -> (?, 1, 128)
            hidden_with_time_axis = tf.manip.tile(
                    hidden_with_time_axis, [1, DEFINES.max_sequence_length, 1])     # (?, 1, 128) -> (?, 25, 128)
            #max_sequence만큼 확장하면서 복제해라.

            # additive attention score
            score = V(tf.nn.tanh(W1(encoder_outputs) + hidden_with_time_axis))      # (?, 25, 1)   #질문과 단어의 연관도?
            #하이퍼볼릭탄젠트 : dot-product attention보다 더 좋다고 제안함.
            attention_weights = tf.nn.softmax(score, axis=-1)                       # (?, 25, 1)   #관련도히스토그램
            #softmax함수로 표준화시킴:
            context_vector = attention_weights * encoder_outputs                    # (?, 25, 128)
            context_vector = tf.reduce_sum(context_vector, axis=1)                  # (?, 25, 128) -> (?, 128)
            input_token_emb = tf.concat([context_vector, input_token_emb], axis=-1) # (?, 256)

            # RNNCell을 호출하여 RNN 스텝 연산을 진행하도록 한다.
            decoder_outputs, decoder_state = rnn_cell(input_token_emb, decoder_state)
            
            # feedforward를 거쳐 output에 대한 logit값을 구한다.
            output_logits = tf.layers.dense(decoder_outputs, params['vocabulary_length'], activation=None)

            # softmax를 통해 단어에 대한 예측 probability를 구한다.
            output_probs = tf.nn.softmax(output_logits)
            output_token = tf.argmax(output_probs, axis=-1)

            # 한 스텝에 나온 토큰과 logit 결과를 저장해둔다.
            predict_tokens.append(output_token)
            temp_logits.append(output_logits)
            
            # PREDICTION 할 때 output_token == <END> 이면 for 문을 break하고 싶은데, 어떻게 하지??
            # tf.cond()로 잘 안되는 거 같은데...

        # 저장했던 토큰과 logit 리스트를 stack을 통해 메트릭스로 만들어 준다.
        predict = tf.transpose(tf.stack(predict_tokens, axis=0), [1, 0])    # shape=(?, 10)
        logits = tf.transpose(tf.stack(temp_logits, axis=0), [1, 0, 2])     # shape=(?, 10, 20705)

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
