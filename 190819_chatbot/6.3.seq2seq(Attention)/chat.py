# Seq2Seq + (Additive) Attention을 이용한 챗봇
#
# 본 프로그램은 "전창욱 등 저, 텐서플로와 머신러닝으로 시작하는 자연어 처리, 위키북스"
# 책에 수록된 챗봇 프로그램을 수정한 버전이다.
#
# Seq2Seq (Basicc) 버전의 Decoder 부분에 attention 기능을 추가했다.
# 
# 사용법 : 
# 학습 : python main.py
# 채팅 : python chat.py
# 결과 :
# Q: 안녕하세요
# A: 안녕하세요 
#
# Q: 가끔 뭐하는지 궁금해
# A: 그 사람도 그럴 거예요 
#
# Q: 남자친구 또 운동 갔어
# A: 운동을 함께 해보세요 
#
# Q: 가스불 켜놓고 나온거 같아
# A: 빨리 집에 돌아가서 끄고 나오세요
#
# Q: quit
#
# 2019.8.17, 아마추어 퀀트 (blog.naver.com/chunjein)
# ----------------------------------------------------------------------------------
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
            'learning_rate': DEFINES.learning_rate, 
            'vocabulary_length': vocabulary_length, 
            'embedding_size': DEFINES.embedding_size
        })

# 확률적 답변을 위해 다항분포 샘플링을 이용한다.
# preds는 디코더의 출력인 softmax이고, beta는 softmax의 강도를 조절하는 변수이다.
# beta가 작아질수록 정적으로 답변하고, beta가 커질수록 동적으로 답변한다 (랜덤 성향).
def pred_indices(preds, beta = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 0.000000001) / beta
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

# 채팅 시작
for i in range(100):
    question = input("Q: ")
    if question == 'quit':
        break
    
    predic_input_enc = data.data_processing([question], word2idx, DEFINES.enc_input)
    predic_output_dec = data.data_processing([""], word2idx, DEFINES.dec_input)
    predic_target_dec = data.data_processing([""], word2idx, DEFINES.dec_target)

    # 매 번 모델을 리빌드하고 checkpoint를 reload하기 때문에 속도가 늦음.
    # Estimator의 특징이므로 다른 방법으로 보완이 필요함.
    predictions = classifier.predict(input_fn=lambda:data.input_fn(
                        predic_input_enc,
                        predic_output_dec,
                        predic_target_dec,
                        1,
                        1))

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
