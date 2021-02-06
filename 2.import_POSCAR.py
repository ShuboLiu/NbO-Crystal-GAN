import pymysql
import pandas as pd
import xlrd
import os
import numpy as np

#从OQMD数据库导入NbO id和entry_id文件
present_dir = os.path.abspath(os.curdir)
file_path = os.path.join(present_dir, "NbO.csv")
data = pd.read_csv(file_path,header=None)
print(data)
id = data.iloc[0,:]
entry_id = data.iloc[1,:]
print(entry_id.shape)
print(id.shape)

POSCAR_dir_path = os.path.join(present_dir, "POSCAR")

#输出该POSCAR文件所包含原子个数
def get_cell_num(file_path):
    with open(file_path,'r') as f:
        total = 0
        for k in range(len(open(file_path,'r').readlines())):
            cur_line_c=f.readline()
            cur_line_c = cur_line_c.strip()
            cur_line_c = cur_line_c.split()
            if k==6:
                cur_line_c = [int(x) for x in cur_line_c]
                for ele in range(0, len(cur_line_c)): 
                    total = total + cur_line_c[ele] 
    return total


output=[]
for i in range(1, len(id)):
    print('We are processing id=',id[i],'entry_id=',entry_id[i])
    id_i=id[i]
    file_name=str(id_i)
    file_dir_path=os.path.join(POSCAR_dir_path, str(entry_id[i]))
    file_path=os.path.join(POSCAR_dir_path, str(entry_id[i]), file_name)
    with open(file_path,'r') as f:
        line=[]
        cell_number = get_cell_num(file_path)
        for j in range(len(open(file_path,'r').readlines())):
            cur_line=f.readline()
            cur_line = cur_line.strip()
            cur_line = cur_line.split()
            if 1 < j < 5:
                line.append(cur_line)
            if j == 6:
                print(cur_line)
            if 7 < j < (8 + cell_number) :
                line.append(cur_line)
        line=np.array(line)
        line=line.flatten()
        output.append(line)

output=pd.DataFrame(output)
output.to_csv("NbO_structure_file.csv", header=None, index=None)
print('All Done')

"""
NbO_structure_file.csv的数据格式为：
1-9列：三个三维晶格适量
10-11列：Nb和O在原胞中分别的数量
12-最后：Nb和O在原胞中的分子坐标，三个一组
"""