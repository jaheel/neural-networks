import numpy as np

class Padding2D:
    @staticmethod
    def padding_data2D(input_data, filter_size, padding = "SAME"):
        """padding matrix data
        
        Padding the specific input matrix with the specific padding pattern

        Parameters
        ----------
        input_data : {array-like, matrix} of shape (in_data_col, in_data_row)

        filter_size : {int-like, scalar}

        padding : {string-like, scalar} default is 'SAME' (optional: 'SAME', 'VALID')

        Returns
        -------
        input_data : {array-like, matrix} input data after padding

        """

        if padding =='VALID':
            return input_data

        pad_add_number = int((filter_size - 1)/2)

        vol = np.zeros(input_data.shape[0])
        for vol_index in range(pad_add_number):
            input_data = np.insert(input_data, 0, values = vol, axis = 1)
            input_data = np.insert(input_data, input_data.shape[1], values = vol, axis = 1)
        
        col = np.zeros(input_data.shape[1])
        for col_index in range(pad_add_number):
            input_data = np.insert(input_data, 0, values = col, axis = 0)
            input_data = np.insert(input_data, input_data.shape[0], values = col, axis = 0)

        return input_data

    @staticmethod
    def padding_data(input_data, filter_size, padding = "SAME"):
        """padding tensor(3-dim) data
        
        padding origin data of multi-channel

        Parameters
        ----------
        input_data : {array-like, tensor(3-dim)} of shape (in_data_col, in_data_row, channel)

        filter_size : {int-like, scalar}

        padding : {string-like, scalar} default is 'SAME' (optional: 'SAME', 'VALID')

        Returns
        -------
        result : {array-like, tensor(3-dim)} input data after padding

        """
        
        result = np.zeros(input_data.shape)   
        for channel_index in range(input_data.shape[2]):
            result[:, :, channel_index] = Padding2D.padding_data2D(input_data = input_data[:, :, channel_index], filter_size = filter_size, padding = padding)

        return result
