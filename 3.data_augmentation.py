import numpy as np 
import pandas as pd 
import ase

def do_translation(orig_structure, number_of_output):
    structures = np.empty((number_of_output, 4, 3), dtype = float)
    for i in range(0, number_of_output):
        delta = np.random.uniform(0, min(orig_structure[0, :]), size = (1, 3))
        structure = orig_structure
        structure = np.array(structure)
        for ii in range(2, len(orig_structure)):
            structure[ii, :] = delta + orig_structure[ii, :]
        structures[i, :, :] = structure
    return structures

def do_rotation(orig_structure):
    structures = np.empty((6, 4, 3), dtype = float)
    structures[0, :, :] = orig_structure[:, [0, 1, 2]]
    structures[1, :, :] = orig_structure[:, [0, 2, 1]]
    structures[2, :, :] = orig_structure[:, [1, 0, 2]]
    structures[3, :, :] = orig_structure[:, [1, 2, 0]]
    structures[4, :, :] = orig_structure[:, [2, 0, 1]]
    structures[5, :, :] = orig_structure[:, [2, 1, 0]]
    return structures

#读入数据并变为三维数组
"""
每个结构都是一个二维数组
73*4*3结构
"""
structure_data = pd.read_csv(r"NbO_structure_file.csv")
struc_data = np.empty((len(structure_data), 4, 3), dtype=float)
for i in range(len(structure_data)):
    print("Reading file No.", i)
    count = 0
    for ii in range(4):
        for iii in range(3):
            struc_data[i, ii, iii] = structure_data.iat[i, count]
            count += 1

## 开始做平移变换
"""
input: 初始结构4*3矩阵（一个）
output：平移结构4*3矩阵（一堆）
ATT: 结构中一般不包含原始结构
"""
num_output_stru_trans = 200
trans_struc_data = np.empty((len(structure_data), num_output_stru_trans, 4, 3), dtype=float)
for i in range(len(struc_data)):
    trans_struc_data[i, :, :, :] = do_translation(struc_data[i, :, :], num_output_stru_trans)

print("Successfully augment data by", num_output_stru_trans, "times via translation")
print("Output matrix after translation has shape of",trans_struc_data.shape)

## 开始做旋转变换
"""
input: 初始结构3*4矩阵（一个）
output：平移结构3*4矩阵（六个）
ATT：结构中一般包含原始结构
"""

rot_struc_data = np.empty((len(structure_data), 6, 4, 3), dtype = float)
for i in range(len(struc_data)):
    rot_struc_data[i, :, :, :] = do_rotation(struc_data[i, :, :])
print("Successfully augment data by", 6, "times via rotation")
print("Output matrix after rotation operation has shape of", rot_struc_data.shape)

final_struc_data = np.empty((len(struc_data), num_output_stru_trans + 6, 4, 3), dtype = float)
for i in range(len(struc_data)):
    final_struc_data[i, :, :, :] = np.concatenate((trans_struc_data[i, :, :, :], rot_struc_data[i, :, :, :]),axis=0)
print("Final output matrix has shape of", final_struc_data.shape)
np.save("3.data_augmentation.npy", final_struc_data)

print(final_struc_data)