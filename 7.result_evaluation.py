import numpy as np
import pandas as pd 
import difflib
import os

#新生成结构导入
npy_name = "./loss_image/fake_imgs_after_post_process.npy"
raw_data = np.load(npy_name,'r')
print("We have raw data of shape", raw_data.shape)

#原本结构导入
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
output_1=output_1.values
output_1=np.array(output_1)

num_atoms = int((output_1.shape[1]-9)/3)
raw_data_original = np.empty([len(output_1), num_atoms+3, 3])
for i in range(len(output_1)):
    raw_data_original[i, 0:3, :] = output_1[i, 0:9].reshape(3,3)
    for atom_index in range(num_atoms):
        raw_data_original[i, atom_index+3, :] = output_1[i, (atom_index*3+6):((atom_index+1)*3+6)]
print("We have raw data from database of shape", raw_data_original.shape)

def molecu_to_car(raw_data):
    # 分子坐标转笛卡尔坐标
    """
    Xcar=Xa*x+Xb*y+Xc*z
    Ycar=Ya*x+Yb*y+Yc*z
    Zcar=Za*x+Zb*y+Zc*z
    """
    num_atoms = raw_data.shape[1]-3
    data = np.empty([raw_data.shape[0], int(num_atoms), 3])
    for i in range(raw_data.shape[0]):
        crystal_matrix = raw_data[i, 0:3, :]
        coordinate = []
        for atom_index in range(num_atoms):
            atom_in_molecule = raw_data[i, atom_index + 3, :]
            coordinate.append(np.dot(crystal_matrix, np.transpose(atom_in_molecule)))
        data[i, :, :] = np.array(coordinate)
    print("The output shape is",data.shape)
    return data

data = molecu_to_car(raw_data)
data_original = molecu_to_car(raw_data_original)

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
    #计算两个向量的距离（二范数）
    diff = atom_1 - atom_2
    distance = np.linalg.norm(diff, ord=2)
    return distance

# 计算两原子键长
bond_len = np.empty([len(data)])
for i in range(len(data)):
    atoms = data[i, :, :]
    for atoms_index in range(data.shape[1]):
        length = calc_distance(atoms[0], atoms[1])
    
    bond_len[i] = length
print(bond_len)

bond_len_original = np.empty([len(data_original)])
for i in range(len(data_original)):
    atoms = data_original[i, :, :]
    for atoms_index in range(data.shape[1]):
        length = calc_distance(atoms[0], atoms[1])
    
    bond_len_original[i] = length
print(bond_len_original)


print(list(set(bond_len).intersection(set(bond_len_original))))
print(list(set(bond_len_original).intersection(set(bond_len))))

for i in range(len(data)):
    if bond_len[i] > min(bond_len_original) and bond_len[i] < max(bond_len_original):
        print("good",i)
    else:
        print(i)
