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
import model as ml
import data
from configs import DEFINES

# 질문과 응답 문장 전체의 단어 목록 dict를 만든다.
word2idx,  idx2word, vocabulary_length = data.load_vocabulary()

# 질문과 응답 문장을 학습 데이터와 시험 데이터로 분리한다.
train_input, train_label, eval_input, eval_label = data.load_data()

# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
train_input_enc = data.data_processing(train_input, word2idx, DEFINES.enc_input)
train_input_dec = data.data_processing(train_label, word2idx, DEFINES.dec_input)
train_target_dec = data.data_processing(train_label, word2idx, DEFINES.dec_target)
	
# 평가 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
eval_input_enc = data.data_processing(eval_input, word2idx, DEFINES.enc_input)
eval_input_dec = data.data_processing(eval_label, word2idx, DEFINES.dec_input)
eval_target_dec = data.data_processing(eval_label, word2idx, DEFINES.dec_target)

# 에스티메이터를 구성한다.
classifier = tf.estimator.Estimator(
        model_fn=ml.model,
        model_dir=DEFINES.check_point_path, 
        params={
            'hidden_size': DEFINES.hidden_size, 
            'learning_rate': DEFINES.learning_rate, 
            'vocabulary_length': vocabulary_length, 
            'embedding_size': DEFINES.embedding_size
        })

# 학습 실행
tf.logging.set_verbosity(tf.logging.INFO)
classifier.train(input_fn=lambda:data.input_fn(
                    train_input_enc,
                    train_input_dec,
                    train_target_dec,
                    DEFINES.batch_size,
                    DEFINES.train_repeats),
                    steps=1000)

# 평가 실행
eval_result = classifier.evaluate(input_fn=lambda:data.input_fn(
                    eval_input_enc,
                    eval_input_dec,
                    eval_target_dec,
                    DEFINES.batch_size,
                    DEFINES.eval_repeats),
                    steps=1)

print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

	

