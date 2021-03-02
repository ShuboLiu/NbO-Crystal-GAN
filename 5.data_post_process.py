import pandas as pd 
import numpy as np 
import xlrd
import os, sys
import shutil

shutil.rmtree("OUTPUT")
os.mkdir("OUTPUT")
raw_data = np.load(r"./loss_image/fake_imgs_gen.npy")
print("We have raw data of shape",raw_data.shape)
"""
数据转换
将晶格常数转换为晶格矢量
"""
data = np.empty((len(raw_data), 3+2, 3), dtype=float)
count = 0
for i in range(len(data)):
    vec_a = np.zeros((3, 1))
    vec_b = np.zeros((3, 1))
    vec_c = np.zeros((3, 1))
    a = raw_data[i, 0, 0]
    b = raw_data[i, 0, 1]
    c = raw_data[i, 0, 2]
    if a<0 or b<0 or c<0:
        print("Not good, will be drop")
    else :
        alpha = raw_data[i, 1, 0]
        beta = raw_data[i, 1, 1]
        gamma = raw_data[i, 1, 2]
        vec_a = [a, 0, 0]
        vec_b = [b*np.cos(gamma), 
                b*np.sin(gamma), 0]
        vec_c = [c*np.cos(beta), 
                c*((np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma)), 
                c*np.sqrt(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-(np.cos(alpha))**2-(np.cos(beta))**2-(np.cos(gamma))**2) / np.sin(gamma)]
                #np.sqrt(c**2 - c**2*np.cos(gamma)**2) - c**2*((np.cos(beta)**2 + np.cos(alpha)**2*np.cos(gamma)**2*c**2 - 2*c**2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))/np.sin(alpha)**2)]
        if np.isnan(vec_c[2]):
            print("Not Good, will be drop")
    
        else:
            data[count, 0, :] = vec_a[:]
            data[count, 1, :] = vec_b[:]
            data[count, 2, :] = vec_c[:]
            for j in range(2, raw_data.shape[1]):
                data[count, j+1, :] = raw_data[i, j, :]
            count += 1

data = data[:count, :]
npy_name="./loss_image/fake_imgs_after_post_process.npy"
np.save(npy_name, data)

"""
此处应该加入数据修正部分代码
1. a b c应该大于0
2. alpha beta gamma应该大于0
"""

## 数据转存为POSCAR文件
print("Shape of potential POSCAR file is", data.shape)
os.makedirs("OUTPUT", exist_ok = True)
for i in range(data.shape[0]):
    out_dir = "NbO_" + str(i)
    cur_path = os.path.abspath(os.curdir)
    out_name=os.path.join(cur_path, "OUTPUT", out_dir)
    with open(out_name, 'w') as f:
        f.write('Nb O \n') 
        f.write('1.0 \n') 
        for j in range(0, 3):
            #f.write(str(data[i, j, :]) + ' \n') 
            f.write("%f %f %f\n" 
                % (data[i, j, 0], data[i, j, 1], data[i, j, 2]))
        f.write('Nb O \n')
        f.write('1 1 \n')
        f.write('Direct \n')
        for j in range(3, 5):
            f.write("%f %f %f\n" 
                % (data[i, j, 0], data[i, j, 1], data[i, j, 2]))

print("All Done")