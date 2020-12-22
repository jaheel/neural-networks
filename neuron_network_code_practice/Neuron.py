import numpy as np
import ActivationFunction as AF
class Neuron(object):
    def __init__(self, weights, bias):
        """
        
        神经元模型

        Parameters
        ----------
        weights : {array-like} of shape(1_sample, n_neuron_layer_before)

        bias : 1_number
        
        """
        self.__weights = weights
        self.__bias = bias
        self.__result = 0

    def feedforward(self, input_data):
        """
        
        前向运算

        Parameters
        ----------
        input_data : {array-like, sparse matrix} of shape(1_samples, n_neuron_layer_before)
        
        Returns
        -------
        result : 1_number
        
        """
        self.__result = np.dot(self.__weights, input_data) + self.__bias
        return self.__result

    def set_weights(self, weights):
        self.__weights = weights
    
    def set_bias(self, bias):
        self.__bias = bias

    def get_weight(self):
        return self.__weights
    
    def get_bias(self):
        return self.__bias
    
    def set_result(self, result):
        self.__result = result


class Layer(object):
    def __init__(self, data, neuron_number, activate_function = "relu"):
        """
        
        单层神经网络

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape(1_samples, n_neuron_layer_before)

    
        neuron_number : (1_number) numbers of neurons in the current layer

        activate_function : 
            1. relu (default)
            2. sigmoid
            3. tanh
            4. leaky_relu
            5. elu
        
        """
        data = np.array(data)

        self.data = data
        self.neuron_number = neuron_number
        self.neurons = [Neuron(weights=self.__weight_init(data.shape[0]), bias=0) for i in range(neuron_number)]
        self.activate_function = activate_function    
        self.AFModel = AF.NNActivator() # 激活函数模型
    
    
    def feedforward(self):
        """

        feed forward

        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        result = []

        for element in self.neurons:
            result.append( element.feedforward(self.data))

        result = self.AFModel.fit(data=result, function_name=self.activate_function)
        
        #将激活后的结果保存到神经元中
        for index in range(len(result)):
            self.neurons[index].set_result(result = result[index])

        return result
    
    def __weight_init(self, in_neuron_numbers):
        """

        小随机数初始化

        Parameters
        ----------
        in_neuron_numbers : (1_number) the number of neurons in the previous layer

         Returns
        -------
        weights : {array-like, np.array } of shape(1_samples, n_in_depth)

        """
        weights = np.random.randn(in_neuron_numbers) / np.sqrt(in_neuron_numbers)
        
        return weights

    def get_weights(self):
        for i in range(self.neuron_number):
            print(self.neurons[i].get_weight())
    
    

        

# weights = np.array([0, 1])
# bias = [4]
# n = Neuron(weights, bias)

# x = np.array([2, 3])
# print(n.feedforward(x))

input_data = np.array([-1,1,2])

layer_1 = Layer(data = input_data, neuron_number=2, activate_function="sigmoid")
result = layer_1.feedforward()
layer_1.get_weights()

layer_2 = Layer(data = result, neuron_number=3,  activate_function="sigmoid")
result = layer_2.feedforward()
layer_2.get_weights()

print(result,"test")