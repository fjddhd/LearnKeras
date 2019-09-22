from keras.models import Sequential
from keras.layers import Dense
import numpy as np
def PimaIndians():
    #设定随机数种子
    np.random.seed(7)
    #导入数据
    dataset=np.loadtxt('../Lib/pima-indians-diabetes.csv',delimiter=',')
    #分割输入变量x和输出变量Y
    #分片：前包后不包，前八列为输入维度，最后一列为输出维度
    x=dataset[:,0:8]
    Y=dataset[:,8]
    #创建模型
    model=Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    #模型编译
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=
                  ['accuracy'])
    #训练模型
    model.fit(x=x,y=Y,epochs=1200,batch_size=10)
    #评估模型
    scores=model.evaluate(x=x,y=Y);
    print('\n%s:%.2f%%'%(model.metrics_names[1],scores[1]*100))