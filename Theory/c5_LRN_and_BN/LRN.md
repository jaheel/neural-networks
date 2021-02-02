# LRN

Local Response Normalization

局部响应归一化

针对一个通道的数据进行归一化操作，公式如下：
$$
b_{x, y}^i = a_{x,y}^i / \bigg{(} k+\alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2 \bigg{)}^ \beta
$$
超参数$(k, \alpha, \beta, n)$

