import numpy as np
import pandas as pd 
import difflib

npy_name = "./loss_image/fake_imgs_after_post_process.npy"
raw_data = np.load(npy_name,'r')
print("We have raw data of shape", raw_data.shape)


# 分子坐标转笛卡尔坐标
"""
Xcar=Xa*x+Xb*y+Xc*z
Ycar=Ya*x+Yb*y+Yc*z
Zcar=Za*x+Zb*y+Zc*z
"""
data = np.empty([raw_data.shape[0], 6])
for i in range(raw_data.shape[0]):
    item = np.empty([1, 6])
    Xa = raw_data[i, 0, 0]
    Xb = raw_data[i, 0, 1]
    Xc = raw_data[i, 0, 2]
    Ya = raw_data[i, 1, 0]
    Yb = raw_data[i, 1, 1]
    Yc = raw_data[i, 1, 2]
    Za = raw_data[i, 2, 0]
    Zb = raw_data[i, 2, 1]
    Zc = raw_data[i, 2, 2]
    item_single = []
    for j in range(raw_data.shape[1]-3): # 典型值为2
        x = np.empty([2,1]); y = np.empty([2,1]); z = np.empty([2,1])
        x[j] = raw_data[i, j+3, 0]
        y[j] = raw_data[i, j+3, 1]
        z[j] = raw_data[i, j+3, 2]
        
        item_x = Xa*x[j] + Xb*y[j] + Xc*z[j]
        item_single.append(item_x)
        item_y = Ya*x[j] + Yb*y[j] + Yc*z[j]
        item_single.append(item_y)
        item_z = Za*x[j] + Zb*y[j] + Zc*z[j]
        item_single.append(item_z)

    item_single = np.array(item_single)
    data[i, :] = item_single.reshape(1, 6)

print(data.shape)

def difflib_leven(str1, str2):
   leven_cost = 0
   s = difflib.SequenceMatcher(None, str1, str2)
   for tag, i1, i2, j1, j2 in s.get_opcodes():
       if tag == 'replace':
           leven_cost += max(i2-i1, j2-j1)
       elif tag == 'insert':
           leven_cost += (j2-j1)
       elif tag == 'delete':
           leven_cost += (i2-i1)
   return leven_cost        

def calc_leven_materix(data):
    # 计算编辑距离程序
    Levenshtein_matrix = np.empty([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            Levenshtein_matrix[i, j] = difflib_leven(data[i], data[j])
    return Levenshtein_matrix

def calc_distance(atom_1, atom_2):
    diff = atom_1 - atom_2
    distance = np.linalg.norm(diff, ord=2)
    return distance

# 计算两原子键长
bond_len = np.empty([len(data)])
for i in range(len(data)):
    atoms = np.empty([int(data.shape[1]/3), 3])
    atoms[0, :] = data[i, 0:3]
    atoms[1, :] = data[i, 3:6]
    length = calc_distance(atoms[0], atoms[1])
    bond_len[i] = length**2
print(bond_len)

