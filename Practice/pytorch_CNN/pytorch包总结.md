# pytorch包总结

## 常用包

torch.utils.data 模块提供了有关数据处理的工具

torch.nn 模块定义了大量神经网络的层

torch.nn.init 模块定义了各种初始化方法

torch.optim 模块提供了模型参数初始化的各种方法



## torchvision

torchvision.datasets : 一些加载数据的函数及常用的数据集接口

torchvision.models : 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等

torchvision.transforms : 常用的图片变换，例如裁剪、旋转等

torchvision.utils : 其他的一些有用的方法



# 模型构造

* 通过继承Module类来构造模型
* Sequential、ModuleList、ModuleDict类都继承自Module类
* 虽然Sequential等类可以使模型构造更加简单，但直接继承Module类可以极大地拓展模型构造的灵活性



