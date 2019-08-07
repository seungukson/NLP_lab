import tensorflow as tf
from data_part import x_pred, y_pred
from model_part import model_fn

# 모델-1로 예측
estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir='./data_out/checkpoint/dnn')
pred_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_pred}, shuffle=False)

for y in estimator.predict(pred_input_fn):
    print("predicted = ", y['result'][0])

print("price sequence = ", x_pred)
print("actual value = ", y_pred)
