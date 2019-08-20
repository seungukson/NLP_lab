import tensorflow as tf
import numpy as np
import data
import model as ml
from configs import DEFINES

tf.logging.set_verbosity(tf.logging.ERROR)
word2idx,  idx2word, vocabulary_length = data.load_vocabulary()

# 에스티메이터 구성
classifier = tf.estimator.Estimator(
        model_fn=ml.model,
        model_dir=DEFINES.check_point_path, 
        params={ 
            'hidden_size': DEFINES.hidden_size, 
            'layer_size': DEFINES.layer_size, 
            'learning_rate': DEFINES.learning_rate, 
            'vocabulary_length': vocabulary_length, 
            'embedding_size': DEFINES.embedding_size
        })

# 확률적 답변을 위해 다항분포 샘플링을 이용한다.
# preds는 디코더의 출력인 softmax이고, beta는 softmax의 강도를 조절하는 변수이다.
# beta가 작아질수록 정적으로 답변하고 (랜덤 성향이 큼), beta가 커질수록 동적으로 답변한다.
def pred_indices(preds, beta = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 0.000000001) / beta
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

# 채팅 시작
for i in range(10):
    question = input("Q: ")
    if question == 'quit':
        break
    
    predic_input_enc = data.data_processing([question], word2idx, DEFINES.enc_input)
    predic_output_dec = data.data_processing([""], word2idx, DEFINES.dec_input)
    predic_target_dec = data.data_processing([""], word2idx, DEFINES.dec_target)

    predictions = classifier.predict(input_fn=lambda:data.input_fn(
                        predic_input_enc,
                        predic_output_dec,
                        predic_target_dec,
                        1,
                        1))#for문을 돌때마다 네트워크 빌드& 체크포인트 적용, 작업을 수행한다->속도저하
    # Estimator의 특성->보완이 필요한 부분

    # 답변 문장에 대한 softmax 확률을 받는다.
    prob = np.array([v['indexs'] for v in predictions])
    prob = np.squeeze(prob)
    
    # 확률적으로 답변 문장의 인덱스를 생성한다.
    words_index = [pred_indices(p, beta = DEFINES.softmax_beta) for p in prob]

    # 답변 문장의 인덱스를 실제 문장으로 변환한다.
    answer = ""
    for word in words_index:
        if word !=0 and word !=2: # PAD = 0, END = 2
            answer += idx2word[word]
            answer += " "
            
    print("A:", answer)
