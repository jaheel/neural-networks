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
        self.__error = 0
        self.__result = 0

    
    def set_weights(self, weights):
        self.__weights = weights
    
    def get_weights(self):
        return self.__weights

    def set_bias(self, bias):
        self.__bias = bias

    def get_bias(self):
        return self.__bias
    
    def set_result(self, result):
        self.__result = result

    def get_result(self):
        return self.__result
    
    def set_error(self, error):
        self.__error = error

    def get_error(self):
        return self.__error

    def update_weights(self, update_weights):
        """

        更新权重
        
        """
        self.__weights += update_weights
    
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

    def backward(self, input_data, grad):
        """

        BP

        Parameters
        ----------
        input_data : {array-like, sparse matrix} of shape(1_samples, n_neuron_layer_before)

        grad : (1_number) real y - predict y

         Returns
        -------
        weights : {array-like, np.array } of shape(1_samples, n_in_depth)

        """
        self._grad_w = np.dot(input_data, grad)
        self._grad_b = np.dot(np.ones(self.input_data.shape[0]), grad)

    


class Layer(object):
    def __init__(self, neuron_number_prev, neuron_number_current, activate_function = "relu"):
        """
        
        单层神经网络

        Parameters
        ----------
        neuron_number_prev : (1_number) the number of neurons in the previous layer
    
        neuron_number_current : (1_number) the number of neurons in the current layer

        activate_function : 
            1. relu (default)
            2. sigmoid
            3. tanh
            4. leaky_relu
            5. elu
        
        """
        

        self.neuron_number_prev = neuron_number_prev
        self.neuron_number_current = neuron_number_current

        # 用python list 保存神经元
        self.neurons = [Neuron(weights=self.__weight_init(neuron_number_prev), bias=0) for i in range(neuron_number_current)]
        self.activate_function = activate_function    
        self.AFModel = AF.NNActivator() # 激活函数模型 

        # 双向链表化
        self.next = None
        self.prev = None
    
    
    def feedforward(self, input_data):
        """

        feed forward

        Returns
        -------
        result : {array-like, python_list } of shape(1_samples, n_activate_result)
        
        """
        input_data = np.array(input_data)
        self.input_data = input_data

        result = []

        for element in self.neurons:
            result.append( element.feedforward(self.input_data))

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
        for i in range(self.neuron_number_current):
            print(self.neurons[i].get_weights())

    
    def error_update(self, error_update_list):
        """

        更新神经元的误差

        Parameters
        ----------
        error_update_list : {array-like, python_list} of shape(1_samples, n_neuron_error)

        """

        for index_cur in range(self.neuron_number_current):
            self.neurons[index_cur].set_error(error_update_list[index_cur])

    
        
class Network(object):
    def __init__(self, input_data, labels, learn_rate = 0.01):
        """

        Parameters
        ----------
        input_data : {array-like, python_list } of shape(1_samples, n_in_depth)

        labels : {array-like, python_list} of shape(1_samples, n_results)

        learn_rate : (1_number) BP algorithm learn rate

        """
        self.input_data = input_data
        self.labels = labels
        self._learn_rate = learn_rate

        self._head = None
        self._tail = None
        
    def is_empty(self):
        """

        判断网络链表是否为空

        """
        return self._head is None

    def length(self):
        """

        网络层数

        """
        cur = self._head
        count = 0

        while cur is not None:
            count += 1
            cur = cur.next
        
        return count
  
    def items(self):
        """

        遍历网络链表

        """
        cur = self._head

        while cur is not None:
            # 返回生成器
            yield cur.item
            # 指针下移
            cur = cur.next
   
    def add_Layer(self, layer):
        """

        添加Layer

        Parameters
        ----------
        layer : simple net layer
        
        """
        node = layer
        
        if self.is_empty():
            self._head = node
            self._tail = node
        else:    
            # 新结点上一级指针指向旧尾部
            node.prev = self._tail
            self._tail.next = node
            self._tail = self._tail.next

    def update(self):
        pass

    def feedforward(self):
        """

        前馈神经网络, 前向运算
        
        """
        out_result = self.input_data

        if self.is_empty():
            pass

        else:
            cur = self._head
            while cur is not None:
                out_result = cur.feedforward(out_result)
                cur = cur.next
        
        self.result = out_result

    def backward(self):
        pass

    def predict(self):
        """

        预测

        """
        self.feedforward()

        while self.cost_function_MSE() > 0.01:
            self.error_update()
            self.weights_update()
            self.feedforward()
            self.print_weights()
            print("----result-----")
            print(self.result)
        

    def print_weights(self):
        cur = self._head
        while cur is not None:
            print("----weights-----")
            cur.get_weights()
            cur = cur.next

    def cost_function_MSE(self):
        """

        计算MSE损失函数 : total_error = 0.5 * (origin_y - predict_y)

        Returns
        -------
        total_error : (1_number)

        """
        mean = self.labels - self.result
        total_error = 0

        for element in mean:
            total_error += 0.5 * element * element
        
        print("----total error-----")
        print(total_error)

        return total_error

    def error_update(self):
        """
        
        每一层神经网络的误差更新

        """
        cur = self._tail
        error_list = self.labels - self.result

        

        # 从后往前，层层递进更新error
        while cur is not None:
            cur.error_update(error_list)
            error_update_list = []
            
            
            for index_pre in range(cur.neuron_number_prev):
                temp_result = 0
                
                for index_cur in range(cur.neuron_number_current):
                   
                    temp_result += cur.neurons[index_cur].get_weights()[index_pre] * error_list[index_cur]
                
                error_update_list.append(temp_result)

            cur = cur.prev
            error_list = error_update_list

    
    def weights_update(self):
        """

        每一层网络，每一个神经元的权值更新

        """
        cur = self._tail
        pre = self._tail.prev
        
        while cur is not None:

            # pre is None 代表cur这一层是第一层神经元
            if pre is None:
                for index_cur in range(cur.neuron_number_current):
                    
                    weights_update_list = []
                    for index_pre in range(len(self.input_data)):
                        
                        # 每一个w的更新: w_new = w_old + mu * error * x_n
                        w_update = self._learn_rate * cur.neurons[index_cur].get_error() * self.input_data[index_pre]
                        
                        weights_update_list.append(w_update)
                    
                    cur.neurons[index_cur].update_weights(weights_update_list)
                
                cur = cur.prev
            
            else:
                for index_cur in range(cur.neuron_number_current):
        
                    weights_update_list = []
                    
                    for index_pre in range(cur.neuron_number_prev):
        
                        # 每一个w的更新: w_new = w_old + mu * error * x_n
                        w_update = self._learn_rate * cur.neurons[index_cur].get_error() * pre.neurons[index_pre].get_result()
                        
                        weights_update_list.append(w_update)
                    
                    cur.neurons[index_cur].update_weights(weights_update_list)
                
                cur = cur.prev
                if cur.prev is None:
                    pre = None
                else:
                    pre = cur.prev
            
            
            

            
            

                 
                    
                
        

        



        
    
    

        


input_data = np.array([-1,1,2])
label = np.array([1])

test_network = Network(input_data=input_data, labels=label)
test_network.add_Layer(Layer(neuron_number_prev=3, neuron_number_current=2, activate_function="sigmoid"))
test_network.add_Layer(Layer(neuron_number_prev=2, neuron_number_current=3, activate_function="sigmoid"))
test_network.add_Layer(Layer(neuron_number_prev=3, neuron_number_current=1, activate_function="sigmoid"))
test_network.predict()
