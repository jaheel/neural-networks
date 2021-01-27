import numpy as np

class cov2D(object):
    def __init__(self, input_data, filter, stride = 1, padding = 1):
        """

        Parameters
        ----------
        input_data : {array-like} of shape (n_col, n_row) origin data, 2D matrix
        
        filter : {array-like} of shape (filter_size, filter_size) filter matrix (3X3) or (5X5)
        
        stride : (1_number)  default is 1
        
        padding : (1_number) dafault is 1

        """
        self.input_data = input_data
        self.filter = filter
        self.stride = stride
        self.padding = padding

    def __cov_single(self, input_data):
        """
    
        A 2D convolution operation

        Paramters
        ---------
        input_data : {array-like} of shape(filter_size, filter_size)

        Returns
        -------
        result : (1_number)
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

        Parameters
        ----------
        pad_add_number : padding times

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
    
        Returns
        -------
        result : {array-like} of shape ( (input_size-filter_size)/stride + 1, (input_size-filter_size)/stride + 1 )
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