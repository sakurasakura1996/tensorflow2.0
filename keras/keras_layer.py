# 1.导入tf.keras
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)

# 2.构建简单模型
# 2.1 模型堆叠   最常见的模型类型是层的堆叠：tf.keras.Sequential 模型
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))    # layers.dense()  意思是一个全连接层,其中有很多参数，inputs:输入网络层的数据 。。。等等
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

# 2.2 网络配置  也就是上面提到的tf.keras.layers.Dense()中的参数配置
# activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
# kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器（均匀分布）。
# 上面的内容再说清楚一点，该参数可以接受一个名称也就是像 kernel_initializer='Glorot uniform'的格式，或者是直接传入一个初始化类的对象，
# Initializer是所有初始化方法的父类，不能直接使用，如果想要定义自己的初始化方法，请继承此类。下面代码每两行效果相同，两种方法都可以
# kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。

layers.Dense(32,activation='relu')
layers.Dense(32,activation=tf.sigmoid)
layers.Dense(32,kernel_initializer='orthogonal')
layers.Dense(32,kernel_initializer=tf.keras.initializers.glorot_normal)
layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l2(0.01))
layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l1(0.01))

# 3 训练评估
# 3.1 设置训练流程
# 构建好模型后，通过调用compile 方法来配置该模型的学习流程：
model = tf.keras.Sequential()
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
# 构建好模型，通过compile方法 配置模型的运行过程
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),    # 定义优化器的内容，这些都可以直接通过keras中的类直接定义，很方便啊
              loss=tf.keras.losses.categorical_crossentropy,   # 定义损失函数的计算  分类交叉熵函数
              metrics=[tf.keras.metrics.categorical_crossentropy]   # 评价函数用于评估当前训练模型的性能。当模型编译后（compile）
              # ，评价函数应该作为 metrics 的参数来输入。metrices可以是多个参数，所以有一个列表包围起来
              )

# 3.2 输入numpy数据
import numpy as np
train_x = np.random.random((1000,72))
train_y = np.random.random((1000,10))

val_x = np.random.random((200,72))
val_y = np.random.random((200,10))

model.fit(train_x,train_y,epochs=10,batch_size=100,validation_data=(val_x,val_y))   # fit函数用于训练模型

# 3.3 tf.data 输入数据
dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))   # 上一步创建的numpy数据并没有输入到tensorflow的结构中
dataset = dataset.batch(32)                                       # 所以还要利用本部分代码，tf.data.Dataset
dataset = dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x,val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()
model.fit(dataset,epochs=10,steps_per_epoch=30,
          validation_data=val_dataset,validation_steps=3)

# 3.4 评估与预测
test_x = np.random.random((1000,72))
test_y = np.random.random((1000,10))
model.evaluate(test_x,test_y,batch_size=32)
test_data = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data,steps=30)

# predict
result = model.predict(test_x,batch_size=32)
print(result)

# 4.构建高级模型
# 4.1 函数式api
# tf.keras.Sequential 模型是简单的堆叠模型，无法表示任意模型，使用keras函数式api可以构建复杂的模型拓扑
# 例如：多输入模型，多输出模型，具有共享层的模型（同一层被调用多次），具有非序列数据流的模型（残差连接）
# 使用函数式 API 构建的模型具有以下特征：
# 层实例可调用并返回张量。 输入张量和输出张量用于定义 tf.keras.Model 实例。
# 此模型的训练方式和 Sequential 模型一样。
input_x = tf.keras.Input(shape=(72,))
hidden1 = layers.Dense(32,activation='relu')(input_x)   # 最后面的括号应该是指定该层的输入来源
hidden2 = layers.Dense(16,activation='relu')(hidden1)
pred = layers.Dense(10,activation='softmax')(hidden2)

model = tf.keras.Model(inputs=input_x,outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=32,epochs=5)

# 4.2 模型子类化
# 通过对 tf.keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。
# 在 init 方法中创建层并将它们设置为类实例的属性。在 call 方法中定义前向传播
class MyModel(tf.keras.Model):
    def __init__(self,num_classes=10):  # 在init方法中定义层，并将层设为子类MyModel的属性（理解为成员变量）
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32,activation='relu')
        self.layer2 = layers.Dense(num_classes,activation='softmax')

    def call(self,inputs):   # 在 call方法中定义前向传播
        h1 = self.layer1(inputs)
        out = self.layer2(h1)    # 实现了两层连接
        return out

    def compute_output_shape(self,input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


model = MyModel(num_classes=10)    # 类的实例化
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=16,epochs=5)

# 4.3自定义层
# 通过对 tf.keras.layers.Layer 进行子类化并实现以下方法来创建自定义层：
# build：创建层的权重。使用 add_weight 方法添加权重。
# call：定义前向传播。
# compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。
# 或者，可以通过实现 get_config 方法和 from_config 类方法序列化层。
class MyLayer(layers.Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim     # 如果要定义成类中的成员变量，就把这个成员变量加到init的参数中
        super(MyLayer, self).__init__(**kwargs)

    def build(self,input_shape):
        shape = tf.TensorShape((input_shape[1],self.output_dim))
        self.kernel = self.add_weight(name='kernel1',shape=shape,
                                       initializer='uniform',trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self,inputs):
        return tf.matmul(inputs,self.kernel)

    def compute_output_shape(self,input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim']=self.output_dim
        return base_config

    @classmethod
    def from_config(cls,config):
        return cls(**config)

model = tf.keras.Sequential(
    [MyLayer(10),layers.Activation('softmax')]
)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=16,epochs=5)


# 4.3 回调
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(train_x,train_y,batch_size=16,epochs=5,
          callbacks=callbacks,validation_data=(val_x,val_y))

# 5 保持和恢复
# 5.1 权重保存
model =tf.keras.Sequential(
    [layers.Dense(64,activation='relu'),
     layers.Dense(10,activation='softmax')]
)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.save_weights('./weights/model')
model.load_weights('./weights/model')
model.save_weights('./model.h5')
model.load_weights('./model.h5')

# 5.2 保存网络结构
# 序列化成json格式
import json
import pprint
json_str = model.to_json()
pprint.pprint(json.load(json_str))
fresh_model = tf.keras.models.model_from_json(json_str)

#保存为yaml模式  # 需要提前安装pyyaml
yaml_str = model.to_yaml()
print(yaml_str)
fresh_model = tf.keras.models.model_from_yaml(yaml_str)

# 5.3 保存整个模型
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(72,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)
model.save('all_model.h5')
model = tf.keras.models.load_model('all_model.h5')

# 6 将keras用于Estimator
# Estimator API 用于针对分布式环境训练模型。它适用于一些行业使用场景，
# 例如用大型数据集进行分布式训练并导出模型以用于生产
model = tf.keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model)