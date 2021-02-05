# LeNet

## Dataset

MNIST

## Architecture

![image-20210205163902447](images/image-20210205163902447.png)



* 输入层：图像大小 **32x32x1**
* 卷积层：filter大小 5x5，深度 6，padding为0，步长为1，输出大小 28x28x6
* 池化层：filter大小 2x2，步长 2，no padding，输出大小 14x14x6
* 卷积层：filter大小 5x5，深度 16，padding为0，步长为1，输出大小10x10x16
* 池化层：filter大小 2x2，步长 2，no padding，输出大小 5x5x16
* Flatten层：将5x5x16矩阵拉成400维向量
* FC层：neuron为120，激活函数tanh
* FC层：neuron为84，激活函数tanh
* 输出层（FC层）：neuron为10，激活函数softmax



## Highlight

1. 定义了CNN基本框架：卷积层+池化层+全连接层
2. 定义了卷积层（局部链接、权值共享）
3. 用Tanh作为非线性激活函数



## Reference

[Gradient-based learning applied to document recognition_1998](https://link.zhihu.com/?target=http%3A//202.116.81.74/cache/7/03/yann.lecun.com/b1a1c4acb57f1b447bfe36e103910875/lecun-01a.pdf)

# AlexNet

## dataset

ImageNet



## Architecture



![image-20210205162620526](images/image-20210205162620526.png)



![image-20210205162853501](images/image-20210205162853501.png)

## Highlight

1. ReLU
   $$
   f(x) =\max(0,x)
   $$

2. 多GPU运行

3. LRN
   $$
   b_{x,y}^i = a_{x,y}^i / \bigg{(} k+\alpha \sum_{j=\max{(0,i-n/2)}}^{\min(N-1,i+n/2)} (a_{x,y}^j)^2 \bigg{)}^\beta
   $$

4. 重叠池化

5. 在全连接层前两层使用Dropout方法，减少过拟合

## Reference

[ImageNet Classification with Deep Convolutional Neural Networks_2012](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)