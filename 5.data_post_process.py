import pandas as pd 
import numpy as np 
import xlrd
import os

raw_data = np.load(r"fake_imgs_gen.npy")
print("We have raw data of shape",raw_data.shape)

"""
此处应该加入数据修正部分代码
"""

## 数据转存为POSCAR文件
print(raw_data.shape[0])
for i in range(raw_data.shape[0]):
    out_dir = "NbO_" + str(i)
    cur_path = os.path.abspath(os.curdir)
    out_name=os.path.join(cur_path, "OUTPUT", out_dir)
    with open(out_name, 'w') as f:
        f.write('Nb O \n') 
        f.write('1.0 \n') 
        for j in range(2, 4):
            f.write(str(raw_data[i, j, :]) + ' \n') 
        f.write('Nb O \n')
        f.write('3 3 \n')
        f.write('Direct \n')
        for j in range(2, 8):
            if abs(sum(raw_data[i, j, :])) < 1000:
                f.write(str(raw_data[i, j, :]) + ' \n')