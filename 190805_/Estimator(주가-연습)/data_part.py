import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv('005930.csv')
data = np.array(df.iloc[:, 5], dtype=np.float32)
data = list((data - data.mean()) / data.std())

nTime = 20
nStep = 3

x_data = []
y_data = []
for i in range(0, len(data) - nTime, nStep):
    x_data.append(data[i: i + nTime])
    y_data.append(data[i + nTime])

# 학습용 데이터
x_train = np.array(x_data)[:-1, :].reshape(-1, nTime, 1)
y_train = np.array(y_data)[:-1].reshape(-1, 1)

# 예측용 데이터
x_pred = np.array(x_data)[-1:, :].reshape(-1, nTime, 1)
y_pred = np.array(y_data)[-1:].reshape(-1, 1)  # 예측 결과 확인용 (참고용임)

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
            .repeat()\
            .batch(64)\
            .make_one_shot_iterator()\
            .get_next()
    return {'x': dataset[0]}, dataset[1]


