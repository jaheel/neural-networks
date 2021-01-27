import numpy as np

class PoolMethod(object):
    def __init__(self, in_data, filter_size=2, stride=2):
        """
        
        池化初始化

        Parameters
        ----------
        in_data : {array-like} of shape(n_col, n_row) origin data 2D 
        
        filter_size : (1_number)
        
        stride : (1_number)
        """
        self.in_data = in_data
        self.filter_size = filter_size
        self.stride = stride

    def __max_pool_single(self, in_data):
        """

        单次最大池化

        Parameters
        ----------
        in_data : {array-like} of shape(filter_size, filter_size) 
        
        Returns
        -------
        result : (1_number) the max result
        """
        if in_data.shape[0] != self.filter_size or in_data.shape[1] != self.filter_size:
            return None
        
        result = 0
        
        for x_index in range(self.filter_size):
            for y_index in range(self.filter_size):
                result = np.maximum(result, in_data[x_index][y_index])
            
        return result
    
    def __average_pool_single(self, in_data):
        """

        单次平均池化

        Parameters
        ----------
        in_data : {array-like} of shape(filter_size, filter_size) 
        
        Returns
        -------
        result : (1_number) the average result
        """
        if in_data.shape[0] != self.filter_size or in_data.shape[1] != self.filter_size:
            return None
        
        result = 0
        
        for x_index in range(self.filter_size):
            for y_index in range(self.filter_size):
                result += in_data[x_index][y_index]
            
        return result/(self.filter_size ** 2)
        
    
    def max_pool(self):
        """

        最大池化

        C : side_length

        C_output = (C_input - C_kernel)/strides + 1

        Returns
        -------
        result : {array-like} of shape((in_data.shape[0]-filter_size)/stride+1, (in_data.shape[1]-filter_size)/stride+1 )
        """
        
        result = []
        filter_number = int(self.filter_size-1/2)

        for y_index in range(0, self.in_data.shape[0] - filter_number, self.stride):
            result_x = []

            for x_index in range(0, self.in_data.shape[1] - filter_number, self.stride):
                max_pool_in_data = self.in_data[y_index : y_index + self.filter_size, x_index : x_index + self.filter_size]
                result_x.append(self.__max_pool_single(in_data = max_pool_in_data))
            
            result.append(result_x)
        
        return np.array(result)

    def average_pool(self):
        """

        平均池化

        C : side_length

        C_output = (C_input - C_kernel)/strides + 1

        Returns
        -------
        result : {array-like} of shape( (in_data.shape[0]-filter_size)/stride+1, (in_data.shape[1]-filter_size)/stride+1 )
        """
        
        result = []
        filter_number = int(self.filter_size-1/2)

        for y_index in range(0, self.in_data.shape[0] - filter_number, self.stride):
            result_x = []

            for x_index in range(0, self.in_data.shape[1] - filter_number, self.stride):
                average_pool_in_data = self.in_data[y_index : y_index + self.filter_size, x_index : x_index + self.filter_size]
                result_x.append(self.__average_pool_single(in_data = average_pool_in_data))
            
            result.append(result_x)
        
        return np.array(result)

    

# ------------
# test : max_pool_single

# test_data = np.array(
#     [
#         [1, 2],
#         [4, 6]
#     ]
# )

# testPool = MaxPool(test_data)
# print(testPool.max_pool_single(in_data = test_data))
# -------------------

# ----------------
# test : max_pool
# test_data = np.array(
#     [
#         [1, 2, 5, 7, 5],
#         [4, 6, 7, 8, 4],
#         [0, 4, 6, 10, 6],
#         [0, 3, 6, 3, 7],
#         [5, 8, 4, 3, 2]
#     ]
# )

# test_data_2 = np.array(
#     [
#         [1, 2, 5, 7],
#         [4, 6, 7, 8],
#         [0, 4, 6, 10],
#         [0, 3, 6, 3]
#     ]
# )

# test_max_pool = MaxPool(in_data=test_data, filter_size=3, stride=2)
# print(test_max_pool.max_pool())
# -------------------

# --------------
# test : average pool

# test_data = np.array(
#     [
#         [1, 2, 5, 7, 5],
#         [4, 6, 7, 8, 4],
#         [0, 4, 6, 10, 6],
#         [0, 3, 6, 3, 7],
#         [5, 8, 4, 3, 2]
#     ]
# )

# test_data_2 = np.array(
#     [
#         [1, 2, 5, 7],
#         [4, 6, 7, 8],
#         [0, 4, 6, 10],
#         [0, 3, 6, 3]
#     ]
# )

# test_average_pool = PoolMethod(in_data=test_data_2, filter_size=2, stride=1)
# print(test_average_pool.average_pool())

# --------------