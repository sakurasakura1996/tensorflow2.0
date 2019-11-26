import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import csv

# 正好用tensorflow练手重写一下kaggle上面的手写数字识别
# 1.load data
train_data = []
train_label = []
openfile = open('./dataset/minist/train.csv','r')
reader = csv.reader(openfile)
for line in reader:
    train_data.append(line)
train_data =np.array(train_data)
train_label = train_data[1:,0]
train_data = train_data[1:,1:]
train_data = np.array(train_data).astype('float')
train_label = np.array(train_label).astype('float')
# print(train_data.shape)
# print(train_label.shape)
test_data = []
openfile = open('./dataset/minist/test.csv','r')
reader = csv.reader(openfile)
for line in reader:
    test_data.append(line)
test_data =np.array(test_data)
test_data = test_data[1:,:]
test_data = np.array(test_data).astype('float')
print(test_data.shape)


# 2.define the model
model = tf.keras.models.Sequential()
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['acc'])
history = model.fit(train_data,train_label,epochs=50,batch_size=200)

pred = model.predict(test_data)
pred= np.argmax(pred,axis=1)

print(history)
print(pred)
print(pred.shape)
plt.plot(history.epoch,history.history.get('loss'))
plt.show()
plt.plot(history.epoch,history.history.get('acc'))
plt.show()
label =['ImageId','Label']
predd = open("./dataset/minist/sub.csv",'w',newline='')
writer = csv.writer(predd)
writer.writerow(label)
length = len(pred)
for i in range(length):
    predict=[]
    predict.append(i+1)
    predict.append(pred[i])
    writer.writerow(predict)
