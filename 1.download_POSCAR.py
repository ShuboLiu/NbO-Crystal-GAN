import os
import re
import wget
import pymysql
import pandas as pd
import time

#从OQMD数据库导出NbO数据
present_dir = os.path.abspath(os.curdir)
file_path = os.path.join(present_dir, "NbO.csv")
data = pd.read_csv(file_path,header=None)
print(data)
id = data.iloc[0,:]
entry_id = data.iloc[1,:]
print(entry_id.shape)
print(id.shape)

#开始下载POSCAR文件
for i in range(1, len(id)):
    id_i=id[i]
    id_int=str(id_i)
    url = 'http://oqmd.org/materials/export/primitive/poscar/'+id_int
    output_file_name = id_int
    present_dir=os.path.abspath(os.curdir)
    out_path=os.path.join(present_dir, "POSCAR", str(entry_id[i]))
    out_name=os.path.join(present_dir, "POSCAR", str(entry_id[i]), output_file_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path) 
        print(out_path + ' 创建成功')    
    if not os.path.exists(out_name):
        command='wget ' + url + ' -O ' + out_name + ' --tries=5 -o wget.log'
        os.system(command)
        #wget.download(url, out_name)
        print(out_name + ' 下载成功')
    size = os.path.getsize(out_name)
    if size == 0:
        os.system('rm '+out_name)
        command='wget ' + url + ' -O ' + out_name + ' --tries=5 -o wget.log'
        os.system(command)
        print(out_name + ' 清除并下载成功')