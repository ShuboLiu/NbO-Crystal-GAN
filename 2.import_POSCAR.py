import pymysql
import pandas as pd
import xlrd
import os
import numpy as np

#从OQMD数据库导入NbO id和entry_id文件
present_dir = os.path.abspath(os.curdir)
file_path = os.path.join(present_dir, "NbO.csv")
data = pd.read_csv(file_path,header=None)
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


output_1=[];num_count=[]
for i in range(0, len(id)):
    print('First step of id=',id[i],'entry_id=',entry_id[i])
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
                num_count.append(cur_line)
            if 7 < j < (8 + cell_number) :
                line.append(cur_line)
        line=np.array(line)
        line=line.flatten()
        output_1.append(line)

output_1=pd.DataFrame(output_1)
num_count = np.array(num_count,dtype=int)
max_A = np.max(num_count[:,0])
max_B = np.max(num_count[:,1])

def cal_angle(x, y):
    #分别计算两向量的模
    #x = x.reshape(3,1)
    y = y.reshape(3,1)
    l_x = np.linalg.norm(x,ord=1)
    l_y = np.linalg.norm(y,ord=1)
    dot_prod = np.dot(x, y)
    cos = dot_prod/(l_x*l_y)
    angle_arc = np.arccos(cos)
    angle_degree = angle_arc*180/np.pi

    return angle_degree


output_length = 6 + max_A*3 + max_B*3
output = np.zeros((len(id), output_length), dtype = float)
a=[];b=[];c=[];alpha=[];beta=[];gama=[]
for i in range(0, len(id)):
    print('Second step of id=',id[i],'entry_id=',entry_id[i])
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
            if j == 2:
                x = np.linalg.norm(cur_line,ord=1)
                line.append(x)
            if j == 3:
                x = np.linalg.norm(cur_line,ord=1)
                line.append(x)
            if j == 4:
                x = np.linalg.norm(cur_line,ord=1)
                line.append(x)
    #计算并保存alpha, beta, gama
    A = output_1.iloc[i, [0, 1, 2]]
    A = A.values
    A = A.astype(float)
    A = A.reshape(1,3)
    B = output_1.iloc[i, [3, 4, 5]]
    B = B.values.astype(float)
    B = B.reshape(1,3)
    C = np.array(output_1.iloc[i, [6, 7, 8]], dtype=float)
    C = C.reshape(1,3)
    alpha = float(cal_angle(A, B))
    line.append(alpha)
    beta = float(cal_angle(B, C))
    line.append(beta)
    gama = float(cal_angle(C, A))
    line.append(gama)
    for jj in range(5):
        output[i, jj] = line[jj]
    
    #计算并保存分子坐标
    ## 第一个原子
    for jj in range(3*num_count[i, 0]):
        output[i, jj+5] = output_1.iat[i, jj+8]
    ## 第二个原子
    for jj in range(3*num_count[i, 1]):
        output[i, jj+14] = output_1.iat[i, jj+8+num_count[i, 0]*3]


output = pd.DataFrame(output)
print(output)
output.to_csv("NbO_structure_file.csv", header=None, index=None)
print('All Done')

"""
NbO_structure_file.csv的数据格式为：
1-6列：三维晶格适量
7-16列：Nb在原胞中的分子坐标，三个一组，用0补齐
17-24最后：O在原胞中的分子坐标，三个一组，用0补齐
"""