# 查看图片
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt


def load_data(path='mnist.npz'):
	f = np.load(path)
	x_train, y_train = f['x_train'], f['y_train']
	x_test, y_test = f['x_test'], f['y_test']
	f.close()
	return (x_train, y_train), (x_test, y_test)
# 从Keras导入Mnist数据集，需要下载
(X_train, y_train), (X_validation, y_validation) = load_data('mnist.npz')

# 显示9张手写数字的图片
for i in range(0, 9):
    plt.subplot(331 + i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

plt.show()