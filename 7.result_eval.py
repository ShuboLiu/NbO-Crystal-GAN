import os
import sys
import numpy as np 
import shutil
import subprocess

np.set_printoptions(threshold=np.inf)

data = np.load(r"./loss_image/fake_imgs_after_post_process.npy")

"""
error size 1: 结构不合理，vasp直接就不算了
error size 2: 
"""
valid_struc = []
for i in range(len(data)):
    dir_name = "./output_vasp/NbO_" + str(i)
    error_size_1 = os.path.getsize("./output_vasp/NbO_0/out")
    if os.path.getsize(dir_name) == error_size_1:
        print("Pass: error 1")
    else :
        valid_struc.append(i)

print(valid_struc)

# Rerun VASP calculation
for i in valid_struc:
    dir_name = "./output_vasp/NbO_" + str(i)
    command = 'mpirun -n 8 vasp_std > out'
    print("Excuiting No.", i)
    #runcmd(command)
    #os.system('cd ' + dir_name + ' && mpirun -n 8 vasp_std > out')
    os.system('ulimit -s unlimited')
    p = subprocess.Popen(
        command,
        shell=True,
        cwd=dir_name,
    )
    try:
        p.wait(1200)
    except subprocess.TimeoutExpired:
        p.kill()