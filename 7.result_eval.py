import os
import sys
import numpy as np 
import shutil
import subprocess

np.set_printoptions(threshold=np.inf)

data = np.load(r"./loss_image/fake_imgs_after_post_process.npy")
print(data.shape)

# original error 2
jump_structure = [13, 29, 36, 40, 44, 56, 58, 62, 64, 72, 74, 76, 77, 85, 87, 88, 89]

"""
error size 1 [0]: internal error in WFINIT: orbitals linearily dependent at random-number initialization
error size 2 [13]: Unreasonable structure, VASP will ignore
error size 3 [1]: scaLAPACK: Routine ZPOTRF ZTRTRI failed! LAPACK: Routine ZTRTRI failed!
    尝试修改晶格常数（一般扩大）;
    尝试修改INCAR中的ALGO到VERY_FAST
error size 4 [6]: REAL_OPT: internal ERROR:           0           2          -1           0
    据说要改变ENCUT可解决，实测不可以，太大没影响；太小变成error 1
"""

error_1 = []; error_2 =[]; error_3 = []; error_4 = []; valid_struc = []
for i in range(len(data)):
    dir_name = "./output_vasp/NbO_" + str(i)
    out_name = os.path.join(dir_name, "out")
    error_size_1 = os.path.getsize("./output_vasp/NbO_0/out")
    error_size_1_2 = os.path.getsize("./output_vasp/NbO_23/out")
    error_size_2 = os.path.getsize("./output_vasp/NbO_13/out")
    error_size_2_2 = os.path.getsize("./output_vasp/NbO_29/out")
    error_size_3 = os.path.getsize("./output_vasp/NbO_1/out")
    error_size_3_2 = os.path.getsize("./output_vasp/NbO_10/out")
    error_size_3_3 = os.path.getsize("./output_vasp/NbO_11/out")
    error_size_4 = os.path.getsize("./output_vasp/NbO_6/out")
    if os.path.getsize(out_name) == error_size_1:
        print("Pass: error 1")
        error_1.append(i)
    elif os.path.getsize(out_name) == error_size_1_2:
        print("Pass: error 1")
        error_1.append(i)
    elif os.path.getsize(out_name) == error_size_2:
        print("Pass: error 2")
        error_2.append(i)
    elif os.path.getsize(out_name) == error_size_2_2:
        print("Pass: error 2")
        error_2.append(i)
    elif os.path.getsize(out_name) == error_size_3:
        print("Pass: error 3")
        error_3.append(i)
    elif os.path.getsize(out_name) == error_size_3_2:
        print("Pass: error 3")
        error_3.append(i)
    elif os.path.getsize(out_name) == error_size_3_3:
        print("Pass: error 3")
        error_3.append(i)
    elif os.path.getsize(out_name) == error_size_4:
        print("Pass: error 4")
        error_4.append(i)
    else :
        valid_struc.append(i)

print("Error_1 have", error_1)
print("Error_2 have", error_2)
print("Error_3 have", error_3)
print("Error_4 have", error_4)
print("Valid structure have",valid_struc)



# Rerun VASP calculation
'''
for i in [3,4,5,9,10,11,12,13,23,24,28,35,43,46,48,51,54,57,58,70,72,77,82,83,90,94]:
    dir_name = "./output_vasp/NbO_" + str(i)
    out_name = os.path.join(dir_name, "out")
    command = 'vasp_std > out'
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
        p.wait(120)
    except subprocess.TimeoutExpired:
        p.kill()
'''

'''
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
'''