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
# 上面的内容再说清楚一点，如果
# kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。