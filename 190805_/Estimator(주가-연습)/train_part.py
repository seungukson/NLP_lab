import tensorflow as tf
from data_part import train_input_fn
from model_part import model_fn

# 모델-1을 사용
estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir='./data_out/checkpoint/dnn')
estimator.train(train_input_fn, steps = 1000)

