import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(100)
tf.random.set_seed(100)

df = pd.read_csv("./Advertising.csv")
df.head()

df = df.drop(columns=['Unnamed: 0'])
df.head()

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

[(train_features_batch, label_batch)] = train_ds.take(1)

print('\nFB, TV, Newspaper batch 데이터:\n',train_features_batch)
print('Sales batch 데이터:',label_batch)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10,input_dim=3),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Dense(1)
])

print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_ds, epochs=100, verbose=1)

loss = model.evaluate(test_X, test_Y, verbose=0)
predictions = model.predict(test_X)
print("Test Data Loss: ", loss)

for i in range(5):
    print("%d 번째 테스트 데이터의 실제값: %f" % (i, test_Y.iloc[i]))
    print("%d 번째 테스트 데이터의 예측값: %f" % (i, predictions[i][0]))



