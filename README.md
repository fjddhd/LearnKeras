# Keras
## 使用Keras构建深度学习模型的步骤：
- 定义模型----创建一个序贯模型并添加配置层
- 编译模型----指定损失函数和优化器，并调用模型的compile（）函数，完成模型编译
- 训练模型----通过调用模型的fit（）函数来训练模型
- 执行预测----调用模型的evaluate（）或predict（）等函数对新数据进行预测
> 序贯模型（Sequential）：多个网络层的线性堆叠
## 多层感知器实例-印第安人糖尿病诊断
1. 导入数据
2. 定义模型
   - 模型被定义为层序列，本例使用三层完全连接的网络结构，Keras中通常用Dense类来定义完全连接的层，Dense参数：神经元数量（unit）、初始化方法（init）、指定激活函数（activation）、input_dim参数指定输入变量数。
3. 编译模型
   - 指定用于评估一组权重的损失函数（loss）、用于搜索网络不同权重的优化器（optimizer），这里使用有效的梯度下降算法Adam作为优化器，由于是分类问题，采用分类准确度作为度量模型的标准。
4. 训练模型
   - epochs参数，对数据集进行固定次数的迭代，并且使用相对较小的batch_size参数，这些参数最好通过多次试验获得。
5. 评估模型
6. 汇总代码
## 多层感知器(Multi-Layer Perceptron，MLP)速成
- 也叫人工神经网络(Artificial Neural Network，ANN)，通称神经网络、多层感知器，常用来解决分类和回归问题，一种前反馈人工神经网络模型，将输入的多个数据维度映射到单一的输出的数据维度上。
- 神经元，是构成神经网络的基本模块，其模型是一个具有加权输入，并且使用激活功能产生输出信号的基础计算单元。
- 权重和偏差：偏差是改善学习速度和预防过拟合的有效方法。权重通常被初始化为小的随机值。
- 激活函数：加权输入与神经元输出的简单映射，它控制神经元激活的阈值和输出信号的强度。
   - 非线性
   - 可微性：优化是基于梯度优化时，必须可微
   - 单调性
   - f（x）≈x
   - 输出值的范围

- 常用的激活函数：类似逻辑函数的非线性函数：sigmoid函数，双曲正切函数：tanh，线性整流函数：ReLU
- 神经网络：
   - 输入层和输出层的节点数往往是固定的，中间层（隐藏层）可以自由指定。
   - 拓扑和箭头代表预测过程中数据的流向，和训练时的数据流有一定的区别。
   - 结构图中的连接线是关键，每个连接线对应一个不同的权重，这需要训练得到。
   - 输入层（可视层）：每个输入维度或列具有一个神经元，作用仅仅是将输入值传递给下一层。
   - 隐藏层：理论证明两层神经网络可以无限逼近任意连续函数，隐藏层对原始数据进行了一个空间变换，使其可以被线性分类。输出层的决策分界画出了一个线性分类分界线，对其进行分类。多层的神经网络的本质就是复杂函数的拟合。
   - 输出层：也称最后的隐藏层，负责输出与项目问题所需格式相对应的值或向量。输出层中激活功能的选择受建模问题类型的强烈约束。
   - 设计一个神经网络时，输入层的节点数需要与特征的维度匹配，输出层的节点数要与目标的维度匹配。
   - 中间层的节点数要根据最终效果来决定。
- 随机梯度下降算法：经典有效的神经网络优化算法，是随机和优化相结合的产物，是用于大规模问题。采用迭代的策略，找到局部最优解，对于凸优化问题，可以得到全局最优解。随机梯度下降可以降低在深度学习的巨大数据量下普通梯度下降的巨大计算开销。其思想是，不直接计算梯度的精确值，而是用梯度的无偏估计来代替梯度，使得数据快速收敛。
   - 向前传播：输入通过神经元的激活函数，一直传递到输出。
   - 反向传播：首先将网络的输出与预期输出进行比较，并计算出错误，然后该错误通过网络一次（依次）传播回来，并根据它们对错误的贡献量更新权重。
   - 时期（epoch）：整个训练数据集的一轮更新网络。
- 权重更新
   - 通常使用少量样本对权重进行更新，也就是说设置一个相对较小的批处理大小（batch_size）
   - 动量（Momentum）
   - 学习率衰减（Learning Rate Decay）

- 预测新数据
   - 单层神经网络中sgn激活函数，两层：sigmoid激活函数，多层神经网络中，ReLU更容易收敛，并且预测性更好。
   - ReLU，不是传统非线性函数，是分段线性函数，表达为y=max（x，0）
   - 在多层神经网络中，训练的主题是优化和泛化。
   - 深度学习中由于层数增加了，参数也增加了，表示能力大幅增强，容易过拟合，正则化很重要。
## 评估深度学习模型
- 两种评估方法：自动评估、手动评估
- 自动评估：fit()函数每次epoch会自动评估，也可以设置validation_split这个参数根据数据集百分比来进行评估，比如设置0.2，就使用20%数据集开始测试。
- 手动评估
   - 手动分离数据集并评估：也可以用scikit机器学习库的train_test_split()函数把数据分成训练数据集和评估数据集。但评估数据集可以通过评估参数传递给Keras的fit()函数。
   - K折交叉验证：sk_Learn中有StratifiedKFord可以用。
##在Keras中使用Scikit_Learn
- 包装类（Wrapper）：Keras中这个类可以将Keras的深度学习模型包装成Scikit_Learn中的分类模型和回归模型，方便使用Scikit_Learn中的方法和函数。
- 使用交叉验证
- 使用网格搜索
## 多分类示例：鸢尾花分类
- 不采用UCI的数据文件，因为是文本数据，如果要使用可以用Pandas的read_csv()函数导入数据，并使用Scikit_Learn的LabelEncoder将类别文本编辑成数值。
- 输出层采用softmax作为激活函数，输出网络预测每个类值的概率，选择具有最高概率的输出。
- Adam作为优化器，在dl模型中用于替代随机梯度下降，默认参数够用
## 回归问题实例：波士顿房价预测
- 采用Scikit_Learn中提供的数据集，十四个特征：
   - CRIM：城镇人均犯罪率
   - ZN：住宅用地所占比例
   - INDUS：城镇中非住宅用地所占比例
   - CHAS：虚拟变量，用于回归分析
   - NOX：环保指数
   - RM：每栋住宅的房间数
   - AGE：1940年以前建成的自住单位的比例
   - DIS：距离5个波士顿的就业中心的加权距离
   - RAD：距离高速公路的便利指数
   - TAX：每一万美元的不动产税率
   - PTRATIO：城镇中教师和学生的比例
   - B：城镇中黑人的比例
   - LSTAT：地区中有多少房东属于低收入人群
   - MEDV：自住房屋房价中位数
- 构建模型参数具有默认值，当成基准模型
- 编译时采用adam优化器，均方误差（MSE）作为损失函数，同时采用相同的均方误差来评估模型的性能，值越小代表模型性能越好。
- 数据预处理：使用Scikit_Learn中的Pipeline
- 调参择优：用Scikit_Learn中的GridSearchCV（）
## 二分类实例：银行营销分类
- 16个输入特征和1个输出特征
- pandas可以对数据进行转换处理
- 用Scikit_Learn库中的函数评估
- Scikit_Learn的StandardScaler对数据进行标准化处理，然后采用十折交叉验证来评估。
- Scikit_Learn的GridSearchCV调参批量测试
## 多层感知器进阶
- 训练神经网络模型需要很久，所以如何对模型进行序列化和反序列化很重要。
- HDF5：模型的权重回报存在HDF5中，模型的结构可以保存在JSON和YAML中。
- 序列化：Keras的to_json()函数生成模型的JSON描述，并保存到JSON文件中。
- 反序列化：通过model_from_json()函数加载JSON描述，编译成模型，可以直接用于生产环境。
- 保存和加载权重：save_weights()保存模型的权重值，并在加载模型时用load_weights()函数加载模型的权重信息。
- 也可以用YAML序列化
- 模型增量更新：对于时效性和计算量的权宜之计
- 神经网络检查点（checkpoint）：长时间运行的容错技术，可以用检查点来驳货模型的权重。Keras中ModelCheckpoint可以定义模型的权重值检查点保存文件的位置、文件的名称，以及在什么情况下创建模型的检查点，使用时需要把检查点放到callback_list中并传给model.fit()的callbacks参数
- 自动保存最优模型：这个检查点策略是把模型权重保存到同一个文件中，当且仅当模型的准确度提高时，才会将权重更新保存到文件中。
- matplotlib的pyplot可以提供数据的可视化。
## Dropout与学习率衰减
- 两类比较严重的问题：
   - 过拟合
   - 时间开销过大
- Dropout：针对神经网络模型的正则化方法，在训练过程中会随机地忽略部分神经元。即：正向传播中：被忽略的神经元对下游神经元的贡献效果暂时消失。反向传播中：这些神经元也不会有权值的更新。它强迫一个神经单元和随机挑选出来的其他神经单元共同工作，减弱了神经元节点间的联合适应性，增强泛化能力。注：隐含节点Dropout率等于0.5时效果最好，也可用于输入层。只能用于模型训练过程，不可用于评估模型阶段。
- 输入层使用Dropout用法（同时创建了输入层）示例：
>     model.add（Dropout（rate=0.2,intput_shape=(4,）） 
>     //导入keras.optimizers的SGD
>     //定义Dropout
>     sgd=SGD(lr=0.1,momentum=0.8,decay=0.0,nesterov=False)
>     //模型编译时在optimizer参数中传入sgd

- 也可以在两层之间直接添加仅带参数rate的Dropout
- Dropout使用技巧：
   - 通常在20%~50%由低到高设置
   - 在较大网络中可能有更好表现
   - 在网络中每一层应用Dropout都显示出良好的效果
   - 使用较高学习率，并使用学习率衰减和巨大的动量值
   - 限制网络权重大小
- 学习率衰减
   - 基本思想：学习率随着训练的进行逐渐衰减。
   - 特点：早期学习速度快，逐步进行微调，直到找到最优值
   - 方法：线性衰减（根据epoch逐步降低学习率）、指数衰减（在特定的epoch是用分数快速降低学习率）
   - 线性学习速度衰减公式：Learning Rate =Learning Rate/(1+decay*epoch)
   - 这几个公式参数可以传递给sgd函数
   - 学习率指数衰减(比如每10次（epochs_drop=10）学习率衰减50%（drop=0.5）)：lrate=init_lrate*poy(drop，floor(1+epoch)/epochs_drop)
- 学习率衰减使用技巧：
   - 提高初始学习率
   - 使用大动量
## 卷积神经网络（Convolutional Neural Networks，CNN）速成
- 一种前馈神经网络，其神经元可以响应一部分覆盖范围内的神经元，并保存了问题的空间结构。适用于计算机视觉和自然语言处理。尤其是避免了对图像的复杂前期预处理，可以直接输入原始图像。
- 基本结构：
   - 特征提取层：每个神经元的输入与前一层的局部接受域相连，并提取该局部的特征。一旦该局部特征被提取后，它与其他特征间的位置关系也随之确定下来。
   - 特征映射层：网络的每个计算层由多个特征映射组成，每个特征映射是一个平面，平面上所有神经元的权值相等。
   - 特征映射结构使用影响函数核小的sigmoid函数，使得特征映射具有位移不变性。
   - 由于一个映射面上的神经元共享权值，减少了网络自由参数的个数。
   - 卷积神经网络中每一个卷积层都紧跟一个用来求局部平均与二次提取的计算层，这种特有的二次特征提取结构减少了特征分辨率。
   - 主要用途：识别位移，缩放及其他形式扭曲不变性的二维图形。
   - 主要特点：多维输入向量可以直接输入网络，避免了特征提取和分类过程中数据重建的复杂度。
   - 保持空间关系：通过使用小的输入数据的平方值，来学习内部特征。
   - 好处都有啥：
      - 比完全连接的网络使用较少的参数（权重）来学习
      - 忽略需要识别的对象在图片中的位置和失真的影响
      - 自动学习和获取输入域的特征
   - 通常包含：
      - 卷积层：用来局部感知提取特征，降低输入参数的层
         - 滤波器（Filter）：本质是该层的神经元，具有加权输入并产生输出值，输入是固定大小的图像。如果卷积层是输入层则输入的是像素值，如果卷积层是中间层，则从前一层的特征图中获得输入。隐含原理：图像的一部分统计特征与其他部分是一样的。
         - 特征图（Feather Map）：由卷积核卷积出来。
      - 池化层：对输入的特征图进行压缩，使特征图变小，简化网络计算复杂度。采用池化层可以忽略目标的倾斜、旋转之类的相对位置的变化，以提高精度，同时降低了特征图的维度，并在一定程度上避免了过拟合。通常以取平均值或最大值来创建自己的特征图。
      - 全连接层：起到“分类器”的作用。如果说卷积层和池化层是将原始数据映射到隐藏层的特征空间的话，这一层是把学到的“分布式特征表示”映射到样本标记空间。通常用非线性激活函数或softmax激活函数。通常在网络末端使用它用于创建特征的最终非线性组合，并用于预测。
## 手写数字识别
- MNIST数据集：在计算机视觉中用于评估手写数字分类问题。
- 多层感应器模型
   - 输入0-255的28*28矩阵（784个0-255的数），输入层和隐藏层都是784个神经元，用ReLU激活，输出层10个神经元，用softmax激活。
   - 输入之后对数据进行归一化处理，即除以255化为0~1之间的数
   - 对输出结果使用keras.utils实例的to_categorical()进行one-hot编码
> **one-hot编码:**使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
> 例如：
> 
> 自然状态码为：000,001,010,011,100,101 
> 
> 独热编码为：000001,000010,000100,001000,010000,100000
> 
> 可以这样理解，对于每一个特征，如果它有m个可能值，那么经过独热编
> 码后，就变成了m个二元特征（如成绩这个特征有好，中，差变成one-
> hot就是100, 010, 001）。并且，这些特征互斥，每次只有一个激
> 活。因此，数据会变成稀疏的。
> 这样做的好处主要有：
> 解决了分类器不好处理属性数据的问题
> 在一定程度上也起到了扩充特征的作用

- 简单cnn模型
   - 输入为28*28的图像
   - 第一个隐藏层用Conv2D卷积层，设定5*5感受野，输出具有32个特征图，用input_shape指定期待输入该层的特征（（1,28,28）是28*28的灰阶图），采用ReLU激活函数。具体使用方法：[https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/#conv2d](https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/#conv2d)
   - 定义一个采用最大值MaxPooling2D的池化层，配置其纵向和横向的采样因子（pool_size）为2*2，表示图片在两个维度均变成原来的一半。具体使用方法：[https://keras-cn.readthedocs.io/en/latest/layers/pooling_layer/#maxpooling2d](https://keras-cn.readthedocs.io/en/latest/layers/pooling_layer/#maxpooling2d)
   - 使用Dropout正则化层，配置参数为0.2，随机排除层中20%的神经元以减少过拟合。
   - 使用一个Flatten层不传入参数，它的输出便于标准的全连接层的处理。
   - 使用ReLU激活函数的128个神经元的全连接层。
   - 输出层具有10个神经元对应着10个分类，采用softmax激活函数，输出美妆图片在每个分类的得分。
   - 模型编译采用categorical_crossentropy损失函数和Adam优化器来编译模型，并采用epochs=10和batch_size=200来训练模型。
- 复杂cnn模型
   - 可以用多个卷积层，具体不谈了
## Keras中的图像增强
- 通过ImageDataGenerator类来实现图像增强处理功能
   - 特征标准化
   - ZCA白化
   - 随机旋转、移动、剪切、翻转图像
   - 维度维持
   - 保存增强后的图像
- 该api被设计成训练过程中实时对数据集进行图像增强，而不是在内存中对整个图像执行操作，以减少内存开销。
- 具体操作不谈了
## 图像识别实例：CIFAR-10分类
- CIFAR-10数据集：包括60000张32*32的彩色图像，50000张用于训练模型、10000张用于评估模型。
- 注意：在使用keras中内置的cifar10数据集时，文件存在用户文件夹下的.keras中，注意如果有了对应的压缩文件则不会重新下载，直接读取同目录下解压文件夹下的文件。
- 注意输入的是彩色的32*32图片，input_shape为（3,32,32），有红、蓝、绿三个维度，范围是0~255，在训练前将其调整到0~1的范围内。
- 输出结果有10个分类，将其one-hot编码，适用于模型的输出。
- 改进模型：池化层采用GlobalAveragePooling
## 情感分析实例：IMDB影评情感分析
- 情感分析是自然语言处理中很重要的方向。
- imdb数据集包含了25000部电影的评价信息。
- 为了便于在模型训练中使用数据集，Keras提供的这个数据集将单词转化成整数，代表了单词在整个数据集中的流行程度。
- 词嵌入（word embeddings）：是一种将词向量化的概念。原理是，单词在高维空间中被编码为实值向量，其中词语之间的相似性意味着向量空间中的接近度。
- Keras通过嵌入层（Embedding）将单词的正整数表示转换为词嵌入。需要指定词汇大小预期的最大数量，以及输出每个词向量的维度，该层只能用作模型中的第一层。
- 嵌入层例子：比如Embedding(5000,32,input_length=500)代表词向量大小为5000（只对数据集前5000个最常用的单词感兴趣），使用32维向量来表示每个单词，构建嵌入层输出。将评论的长度限制在500个单词以内，长度超过500个单词的将转化为比0更短的值。
- 卷积神经网络被设计为符合图像数据的空间结构，对场景中学习对象的位置和方向是鲁棒的。该原则也可以用于处理序列问题。
## 循环神经网络(RNNs)
- 用来处理序列数据的神经网络，因为一个序列当前的输出与前面的输出有关，每层的节点之间也是有连接的。具体表现为：网络会对前面的信息进行记忆，并应用于当前输出的计算中。
- 这种隐藏层的输入不仅仅包括输入层的输入，还有上一时刻隐藏层的输出。
- 理论上，RNNs具有循环链接，随着时间的推移向网络增加反馈和记忆。这种记忆能力增强了循环神经网络对序列问题的网络学习和泛化输入能力。
- 长短期记忆网络（LSTM）在nlp和序列学习问题中效果良好。
- 序列问题：
   - 涉及到序列进行输入或输出，需要处理其中的相关性，比如股票价格预测等
   - 序列问题分类
      - 一对多：序列输出，用于图像字幕
      - 多对一：序列输入：用于情感分类
      - 多对多：序列输入和输出：用于机器翻译
      - 同步多对多：同步序列输入和输出，用于视频分类
- 循环神经网络（RNNs）：是在一个标准的多层感知器的架构上增加循环链接。比如在给定的层中，每个神经元可以向其最近（侧向）的神经元传递信号，而不是只向前（下一层）传递信号，网络的输出可以作为下一次输入的输入向量反馈给网络。
   - 需要解决两个主要问题：
      - 如何训练具有反向传播的神经网络
      - 如何解决训练过程中梯度消失或梯度爆炸问题
   - 原因：在RNNs中由于神经元的重复连接或循环连接，反向传播存在不适用性，不能很好地完成网络权重的更新。需要使用时间反向传播（BPTT）来解决。即：具有自身连接的单个神经元，展开网络结构，可以表示为具有相同权重值的多个神经元，这样就可以把神经网络的循环图变成经典前馈神经网络的非循环图，并且可以使用反向传播。
   - 解决在反向传播用于非常深的神经网络和RNNs时的梯度爆炸（梯度变成非常大的数值）或梯度消失（梯度变非常小的数值），可以使用激活函数甚至无监督的预训练层来缓解，也可以使用长短期记忆网络（LSTM）来缓解。
- 长短期神经网络（LSTM）：使用时间反向传播（BPTT）训练，解决了梯度消失问题。适合解决比较复杂的序列问题，并具有很高的效率。使用**存储单元**代替常规的神经元，每个存储单元由输入门、输出门和自有状态构成，根据输入序列来操作每一个存储单元，存储单元内的每个门使用sigmoid函数来控制它们是否被触发。
   - 三种类型的门
      - 遗忘门：有条件地决定哪些信息从单元中抛弃
      - 输入门：有条件地决定在单元中存储哪些信息
      - 输出门：有条件地决定哪些信息需要被输出，并输出信息
                                                       