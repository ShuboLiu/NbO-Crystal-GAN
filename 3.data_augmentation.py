import numpy as np 
import pandas as pd 
import ase

def do_translation(orig_structure, number_of_output):
    print("Doing translation")
    structures = np.zeros(((number_of_output, 3, 8)))
    for i in range(1, number_of_output):
        delta = np.random.uniform(0, 1, size = 3).reshape(1, 3)
        #structure = delta + orig_structure
        structures[i, :, :] = structure
    return structures

#读入数据并变为三维数组
"""
每个结构都是一个二维数组
73*3*8结构
"""
strcture_data = pd.read_csv(r"NbO_structure_file.csv")
struc_data = np.zeros(((73, 8, 3)))
for i in range(len(strcture_data)):
    print("Reading file No.", i)
    count = 0
    for ii in range(7):
        for iii in range(2):
            struc_data[i, ii, iii] = strcture_data.iat[i, count]
            count += 1

print(struc_data)

## 开始做平移变换
"""
input: 初始结构3*8矩阵（一个）
output：平移结构3*8矩阵（一堆）
"""
num_output_stru = 200
for i in range(len(struc_data)):
    trans_struc_data = do_translation(struc_data[i, :, :], num_output_stru)
