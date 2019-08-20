import tensorflow as tf
import model as ml
import data
from configs import DEFINES

# 질문과 응답 문장 전체의 단어 목록 dict를 만든다.
word2idx,  idx2word, vocabulary_length = data.load_vocabulary()
#
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
            'layer_size': DEFINES.layer_size, 
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
                    steps=DEFINES.train_steps)

# 평가 실행
eval_result = classifier.evaluate(input_fn=lambda:data.input_fn(
                    eval_input_enc,
                    eval_input_dec,
                    eval_target_dec,
                    DEFINES.batch_size,
                    DEFINES.eval_repeats),
                    steps=1)

print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

	

