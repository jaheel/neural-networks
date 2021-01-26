import numpy as np

class cov2D(object):
    def __init__(self, input_data, filter, stride = 1, padding = 1):
        """

        @params input_data : origin data, 2D matrix
        @params filter : filter matrix (3X3) or (5X5)
        @params stride : (default: 1)
        @params padding : (dafault: 1)

        """
        self.input_data = input_data
        self.filter = filter
        self.stride = stride
        self.padding = padding

    def __cov_single(self, input_data):
        """
    
        A 2D convolution operation

        @params input_data : {np.array, [n,n](same to filter.shape)} 

        @return result : 1_number
        """
        if input_data.shape[0] != self.filter.shape[0] or input_data.shape[1] != self.filter.shape[1]:
            pass
        
        result = 0

        for shape_0_index in range(input_data.shape[0]):
            for shape_1_index in range(input_data.shape[1]):
                result += input_data[shape_0_index][shape_1_index] * self.filter[shape_0_index][shape_1_index]

        return result
    
    def __padding_data(self, pad_add_number):
        """

        padding data

        @params pad_add_number : padding times

        @params input_data : after padding

        """
        vol = np.zeros(self.input_data.shape[0])
        for vol_index in range(pad_add_number):
            self.input_data = np.insert(self.input_data, 0, values = vol, axis = 1)
            self.input_data = np.insert(self.input_data, self.input_data.shape[1], values = vol, axis = 1)
        
        col = np.zeros(self.input_data.shape[1])
        for col_index in range(pad_add_number):
            self.input_data = np.insert(self.input_data, 0, values = col, axis = 0)
            self.input_data = np.insert(self.input_data, self.input_data.shape[0], values = col, axis = 0)
    
    def convolution(self):
        """

        卷积运算，输出矩阵
    
        

        @return result : np.array 2D
        """


        filter_number = int((self.filter.shape[0] - 1)/2)

        if self.padding == 1:
            self.__padding_data(pad_add_number = filter_number)
    
        else:
            pass
    
        result = []

        for y_index in range(filter_number, self.input_data.shape[0]-filter_number, self.stride):
            result_x = []
            
            for x_index in range(filter_number, self.input_data.shape[1]-filter_number, self.stride):
                input_matrix = self.input_data[y_index - filter_number : y_index + filter_number + 1, x_index - filter_number : x_index + filter_number + 1]
                result_x.append(self.__cov_single(input_data = input_matrix))

            result.append(result_x)

        return np.array(result)


# -------------------
# test code:

# input_data = np.array(
#     [
#         [1, 3, 5, 4, 7],
#         [2, 3, 2, 1, 0],
#         [7, 8, 1, 2, 3],
#         [3, 2, 9, 8, 7],
#         [2, 3, 4, 0, 2]
#     ]
# )

# cov_filter = np.array(
#     [
#         [0, 0, 0],
#         [0, 1, 0],
#         [0, 0, 0]
#     ]
# )
# test_cov = cov2D(input_data=input_data, filter=cov_filter,stride=2, padding=1)
# print(test_cov.convolution())

# -------------------