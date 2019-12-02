# 我们将以mlp为基础模型，然后介绍一些深度学习常见技巧， 如： 权重初始化， 激活函数，
# 优化器， 批规范化， dropout，模型集成
import tensorflow as tf
from tensorflow.keras import layers
from keras import datasets


# 1.导入数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


# 2.基础模型
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train,y_train,batch_size=265,epochs=100,validation_split=.03,verbose=0)
print(history.history)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'],loc='upper left')
plt.show()


result = model.evaluate(x_test,y_test)