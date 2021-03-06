# 数据预处理

主要针对神经网络的输入层数据进行逐属性规范化的过程。

## 中心化

使得单个属性均值为0

## 规范化

使得单个属性的值变化在范围[0,1]或[-1,1]

## PCA

数据降维，选取线性无关性最强的几个属性

## 白化

PCA对数据点云进行了平移和旋转，但形状未变。

利用属性标准差进行规范化，即属性数值除以对应的标准差

几何意义：使每个属性的方差相同，数据点云完全处于高维球内。

```python
X_white = X_decor/np.sqrt(S + 10 ** (-5))
```



# BN(Batch Normalization)

对隐含层输入数据进行规范化的方法。



数据经过网络的多层变换后（其中包括非线性变换），不再使规范化的，即均值不为0，方差不为1。

> 经过ReLU变换后，输出属性都大于或等于0，均值肯定会大于0，这样对隐含层来说，学习就变困难了。



## BN前向计算

操作方式与数据预处理中的中心化和规范化操作一样，都是逐属性进行的，先减去均值，再除以标准差。

公式：
$$
\hat{x} = \frac{x-\mu}{\sigma}
$$

> 数据预处理时：均值和标准差时整个训练集的均值和标准差
>
> BN：批量的均值和标准差（即在使用批量梯度下降法时，每次迭代的样本子集）



缺点：

1. 网络表达能力受限。不管网络前面的层对输入数据进行怎样的变换，最后都变为均值为0、方差为1的分布。（希望不同的层能学到不同的均值和方差）

解决方案：引入两个可学习的参数$\gamma$和$\beta$来学习标准差和均值。

结果，BN公式为如下：
$$
\hat{x} = \frac{x-\mu}{\sigma} \\
y = \gamma \hat{x} + \beta
$$

* $\gamma$初始化为1，$\beta$初始化为0。BN层输出数据的均值为$\beta$，标准差为$\gamma$，与网络前面的层无关，减小了网络中各层的耦合，有利于学习。
* $\gamma$和$\beta$是可学习的，增加了网络的自适应性。



## BN层位置

通常位于非线性激活层之前、全连接层之后，注意必须在激活前对数据进行规范化。

加入BN层的典型网络结构如下：
$$
X=UW \\
Y=BN(X; \gamma, \beta) \\
Z = f(Y)
$$
其中，U是输入数据。

X是全连接层的输出，也是BN层的输入

Y是BN层的输出

最后进行非线性激活，得到输出Z



## BN层的梯度反向传播

前向计算是已知输入X_batch，求输出Y

反向传播是已知dY，求dX_batch、dgamma和dbeta



# 数据扩增

常见数据扩增技术包括：

* 图像进行平移、旋转、左右翻转来分别实现平移、旋转和镜像不变性
* 随即裁剪和缩放实现尺度不变性
* 进行仿射变换和弹性变形实现形变不变性
* 进行模糊、加噪、PCA Jittering和Color Jittering实现色彩不变性



如果把样本看作高维空间的点，那么对样本做数据扩增，就是在样本点的“邻域”内生成新的样本点，达到增加样本数量的目的。（“邻域”不一定是指欧氏距离近，而是“语义”距离近）



PS：数据扩增技术一般只在模型参数数量巨大，而样本数量有限的情况下使用，特别是在深度学习中用来防止过拟合。





# 梯度检查

​		利用梯度下降进行优化时，需要采用链式法则计算损失函数对参数的梯度，这是一个十分容易出错的地方，所以必须对梯度计算进行检查，以确保正确。

​		梯度检查总原则：比较数值梯度和解析梯度值是否一致，其中解析梯度值是采用链式法则计算的梯度值，数值梯度是采用梯度定义计算的梯度值。

​		对参数$w$（标量）的梯度进行检查，损失函数$f(w)$，数值梯度定义：
$$
f_n' = \frac{f(w+h)-f(w)}{h}
$$
​		其中$h$是步长，是小常数，实践中常取$10**(-5)$。

​		计算数值梯度时，需要进行两次前向计算，效率十分低下。实际中采用中心差分法计算梯度：
$$
f_n'= \frac{f(w+h)-f(w-h)}{2h}
$$


# 初始损失值检查

​		梯度检查即使正确，也不能说明代码正确实现了模型，因为梯度检查只是验证了链式法则正确计算了损失函数的梯度而已。模型对不对，还需要进一步检查。

​		最简单的检查办法：采用小的权重对模型进行初始化，关闭正则化，只检查数据损失，此时样本数据最好采用标准正态随机数进行模拟。



# 过拟合微小数据子集

在对整个数据集训练之前，首先对微小数据子集（比如每类2到4个样本）进行训练，争取获得很小的损失值（发生过拟合），此时也要关闭正则化。

* 如果没有获得很小的损失值，则最好不要对整个训练集进行训练。

* 即使模型完全拟合了微小数据子集，也不能保证程序一定正确。

  > 因为模型相对于微小子集容量过大，所以当程序没有正确实现时，只要模型流程是正确的，梯度计算正确，模型就很容易过拟合微小子集。



典型错误有：

1. 忘了加激活层
2. 最后一个全连接层后加了激活层
3. 权重初始化不正确
4. 权重更新错误
5. 数据预处理错误



# 监测学习过程

主要目的：观察超参数是否设置合理。

最重要的超参数需要监控：学习率和正则化强度。

## 损失值

监测损失值随训练周期的变化情况来判断学习率的大小是否合适。

1. 损失值突然增大，大于初始损失值，而且持续增大，这说明学习率太大了。
2. 损失值减小速度很快或者波动太大，很快就饱和了，不再减小，但最终损失值过大，说明学习率比较大。
3. 损失值减小速度很慢，说明学习率过小。
4. 损失值减小速度适中，最终损失值很小，说明学习率比较合适。



## 训练集和验证集的准确率

监测训练集和验证集的准确率差异随训练周期的变化情况，来判断正则化强度是否合适。

* 验证集的准确率远差于训练集的准确率，过拟合
* 验证集的准确率稍差于训练集的准确率，过拟合比较弱
* 验证集的准确率和训练集的准确率一致，说明基本没有过拟合



## 参数更新比例

监测参数的更新比例，可以掌握每层参数的学习情况。

参数更新比例时计算一层中所有参数更新量和参数的比值，注意，需要每层单独监控。

```python
update_param = -lr * dparam
update_ratio = np.sum(np.abs(update_param))/np.sum(np.abs(param))
```

> param是一层的权重或偏置参数
>
> dparam是梯度
>
> update_param是参数更新量
>
> update_ratio是更新比例