import numpy as np
from padding import Padding2D

class ConvLayer(object):
    def __init__(self, filters, kernel_size, input_shape, strides, padding="SAME", activation="None", name="conv"):
        """

        Parameters
        ----------
        filters : (1_number) the number of filters
        
        kernel_size : {tuple-like} of shape (filter_col, filter_row), the size of single filter

        input_shape : {tuple-like} of shape (in_data_col, in_data_row, channel) ep: (64,64,3)

        strides : {tuple-like} of shape (col_stride, row_stride) (ep: (1,1))

        padding : (1_str) padding patern, value of { 'SAME'(default), 'VALID'}

        activation : (1_str) 

        name : (1_str) current layer name
         
        """
        
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__strides = strides
        self.__padding = padding
        self.activation_name = activation
        self.__input_shape = input_shape
        self.__input_padding_shape = input_shape
        self.__input = np.zeros(self.__input_shape)
        self.name = name
        self.flag = False

    
    def __padding_data(self, X):
        """
        
        Parameters
        ----------
        X : {array-like} of shape (in_data_col, in_data_row, channel)

        Returns
        -------
        result : after padding
        """

        result = Padding2D.padding_data(input_data = X, filter_size = self.__kernel_size[0], padding = self.__padding)
        
        return result