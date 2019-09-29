# 简单卷积神经网络

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend
backend.set_image_data_format('channels_first')

# 设定随机种子
seed = 7
np.random.seed(seed)
# 导入数据，文件存在用户文件夹下的.keras中，注意如果有了对应的压缩文件则不会重新下载，直接读取同目录下解压文件夹下的文件
(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()
# 设定格式float
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
# 将数据规整到0~1之间
X_train = X_train / 255.0
X_validation = X_validation / 255.0
print(X_train.shape)
# 类别进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
print(y_train.shape)   # (50000, 1, 10)

def create_model(epochs=25):
	model = Sequential()
	# 卷积层
	model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	# Dropout
	model.add(Dropout(0.2))
	# 卷积层
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	lrate = 0.01
	decay = lrate / epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

epochs = 25
model = create_model(epochs)
model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, verbose=2)
scores = model.evaluate(x=X_validation, y=y_validation, verbose=0)
print('Accuracy: %.2f%%' % scores[1] * 100)
