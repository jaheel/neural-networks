import numpy as np
from padding import Padding2D

input_data_test = np.array(
    [
        [1, 3, 5, 4, 7],
        [2, 3, 2, 1, 0],
        [7, 8, 1, 2, 3],
        [3, 2, 9, 8, 7],
        [2, 3, 4, 0, 2]
    ]
)

print(Padding2D.padding_data2D(input_data = input_data_test, filter_size = 1))