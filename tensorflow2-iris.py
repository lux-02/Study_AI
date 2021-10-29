import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

np.random.seed(100)
tf.random.set_seed(100)

X, Y = load_iris(return_X_y = True)

df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=4),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_ds, epochs=100, verbose=2)

loss, acc = model.evaluate(test_X, test_Y)

predictions = model.predict(test_X)

print("테스트 데이터의 Accuracy 값: ", acc)

for i in range(5):
    print("%d 번째 테스트 데이터의 실제값: %d" % (i, test_Y.iloc[i]))
    print("%d 번째 테스트 데이터의 예측값: %d" % (i, np.argmax(predictions[i])))

