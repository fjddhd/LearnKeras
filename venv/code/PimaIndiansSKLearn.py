from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

def PimaIndiansSKLearn():
    #构建模型
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # 模型编译
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=
    ['accuracy'])
    return model
# 设定随机数种子
seed=7
np.random.seed(seed)
#导入数据
dataset=np.loadtxt('../Lib/pima-indians-diabetes.csv',delimiter=',')
#分割输入变量x和输出变量Y
#分片：前包后不包，前八列为输入维度，最后一列为输出维度
x=dataset[:,0:8]
Y=dataset[:,8]

#创建模型 for scikit_learn  -ATTENTION:参数指定的方法（函数）不需要加括号
model=KerasClassifier(build_fn=PimaIndiansSKLearn,epochs=150,batch_size=10,verbose=0)

# 10折交叉验证
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,x,Y,cv=kfold)
print(results.mean())