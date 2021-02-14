import pandas as pd 
import numpy as np 
import xlrd
import os, sys
import shutil

shutil.rmtree("OUTPUT")

raw_data = np.load(r"fake_imgs_gen.npy")
print("We have raw data of shape",raw_data.shape)
print(raw_data)
"""
数据转换
将晶格常数转换为晶格矢量
"""
data = np.zeros((((len(raw_data), 9, 3))))
for i in range(len(data)):
    vec_a = np.zeros((3, 1))
    vec_b = np.zeros((3, 1))
    vec_c = np.zeros((3, 1))
    a = raw_data[i, 0, 0]
    b = raw_data[i, 0, 1]
    c = raw_data[i, 0, 2]
    alpha = raw_data[i, 1, 0]
    beta = raw_data[i, 1, 1]
    gamma = raw_data[i, 1, 2]
    vec_a = [a, 0, 0]
    vec_b = [b*np.cos(alpha), b*np.sin(alpha), 0]
    vec_c = [c*np.cos(gamma), 
             c*((np.cos(beta) - np.cos(alpha)*np.cos(gamma))/np.sin(alpha)), 
             np.sqrt(c**2 - c**2*np.cos(gamma)**2) - c**2*((np.cos(beta)**2 + np.cos(alpha)**2*np.cos(gamma)**2*c**2 - 2*c**2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))/np.sin(alpha)**2)]
    data[i, 0, :] = vec_a[:]
    data[i, 1, :] = vec_b[:]
    data[i, 2, :] = vec_c[:]

    for j in range(2, raw_data.shape[1]):
        data[i, j+1, :] = raw_data[i, j, :]

"""
此处应该加入数据修正部分代码
1. a b c应该大于0
2. alpha beta gamma应该大于0
"""

## 数据转存为POSCAR文件
print("Shape of potential POSCAR file is", data.shape)
os.mkdir("OUTPUT")
for i in range(data.shape[0]):
    out_dir = "NbO_" + str(i)
    cur_path = os.path.abspath(os.curdir)
    out_name=os.path.join(cur_path, "OUTPUT", out_dir)
    with open(out_name, 'w') as f:
        f.write('Nb O \n') 
        f.write('1.0 \n') 
        for j in range(0, 3):
            f.write(str(data[i, j, :]) + ' \n') 
        f.write('Nb O \n')
        f.write('3 3 \n')
        f.write('Direct \n')
        for j in range(3, 9):
            f.write(str(data[i, j, :]) + ' \n')