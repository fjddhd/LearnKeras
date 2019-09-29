#这里使用了toimage的替代方法：PIL的Image
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from PIL import Image
# from scipy.misc import toimage
import numpy as np
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# def load_data(path='mnist.npz'):
# 	f = np.load(path)
# 	x_train, y_train = f['x_train'], f['y_train']
# 	x_test, y_test = f['x_test'], f['y_test']
# 	f.close()
# 	return (x_train, y_train), (x_test, y_test)
# # 从Keras导入
# (X_train, y_train), (X_validation, y_validation) = unpickle('D:\cifar-10-batches-py\data_batch_1')
#两个元组
(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()
print(type((X_train, y_train)))
print(type(X_train))
# 展示9张图片样例
for i in range(0, 9):
	plt.subplot(331 + i)
	plt.imshow(Image.fromarray(X_train[i]))
plt.show()
