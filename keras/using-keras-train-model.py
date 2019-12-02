# 使用keras来训练模型
# 1.一般的模型构造/训练/测试流程
# 模型构造
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import keras
inputs = keras.Input(shape=(784,),name='mnist_input')
h1 = layers.Dense(64,activation='relu')(inputs)
h1 = layers.Dense(64,activation='relu')(h1)
outputs = layers.Dense(10,activation='softmax')(h1)
model = keras.Model(inputs,outputs)

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.sparse_categorical_crossentropy(),
              metrics=[keras.metrics.sparse_categorical_accuracy()])

#载入数据
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 训练模型
history = model.fit(x_train,y_train,batch_size=64,epochs=3,validation_data=(x_val,y_val))
print('history:')
print(history.history)

result = model.evaluate(x_test,y_test,batch_size=128)

print('evaluate:')
print(result)

pred = model.predict(x_test[:2])
print('predict:')
print(pred)

#2.自定义损失和指标
# 自定义指标只需要继承Metric类，并重写一下函数
# _init_(self)，初始化
