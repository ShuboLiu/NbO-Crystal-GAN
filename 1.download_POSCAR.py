import os
import re
import wget
import pymysql
import pandas as pd
import time
import subprocess

#从OQMD数据库导出NbO数据
present_dir = os.path.abspath(os.curdir)
file_path = os.path.join(present_dir, "NbO.csv")
## 第一行id，第二行entry_id
data = pd.read_csv(file_path, header=None)
id = data.iloc[0,:]
entry_id = data.iloc[1,:]
print(entry_id.shape)
print(id.shape)

#开始下载POSCAR文件
for i in range(1, len(id)):
    id_i=id[i]
    id_int=str(int(id_i))
    url = 'http://oqmd.org/materials/export/primitive/poscar/'+id_int
    output_file_name = id_int
    present_dir=os.path.abspath(os.curdir)
    out_path=os.path.join(present_dir, "POSCAR", str(int(entry_id[i])))
    out_name=os.path.join(present_dir, "POSCAR", str(int(entry_id[i])), output_file_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path) 
        print(out_path + ' 创建成功')    
    if not os.path.exists(out_name):
        command='wget ' + url + ' -O ' + id_int #+ ' --tries=5 -o wget.log'
        args = [r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe", command]
        #os.system(command) ## 用于linux主机
        print(out_path)
        subprocess.run(args, shell=False, cwd=out_path)
        time.sleep(1)
        print(out_name + ' 下载成功')
    size = os.path.getsize(out_name)
    if size == 0:
        os.system('rm '+out_name)
        command='wget ' + url + ' -O ' + out_name #+ ' --tries=5 -o wget.log'
        args = [r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe", command]
        subprocess.run(args, shell=False, cwd=out_path)
        #os.system(command) ## 用于linux主机
        #wget.download(url, out_name)
        print(out_name + ' 清除并下载成功')