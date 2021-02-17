import os
import sys
import numpy as np 
import shutil
import subprocess

np.set_printoptions(threshold=np.inf)

data = np.load(r"./loss_image/fake_imgs_after_post_process.npy")
print(data.shape)

"""
error size 1 [0]: internal error in WFINIT: orbitals linearily dependent at random-number initialization
error size 2 [0]: 结构不合理，vasp直接就不算了
error size 3 [1]: scaLAPACK: Routine ZPOTRF ZTRTRI failed! LAPACK: Routine ZTRTRI failed!
error size 4 [6]: REAL_OPT: internal ERROR:           0           2          -1           0
"""
error_1 = []; error_2 =[]; error_3 = []; error_4 = []; valid_struc = []
for i in range(93, len(data)):
    dir_name = "./output_vasp/NbO_" + str(i)
    error_size_1 = os.path.getsize("./output_vasp/NbO_0/out")
    error_size_2 = os.path.getsize("./output_vasp/NbO_0/out")
    error_size_3 = os.path.getsize("./output_vasp/NbO_1/out")
    error_size_4 = os.path.getsize("./output_vasp/NbO_6/out")
    if os.path.getsize(dir_name) == error_size_1:
        print("Pass: error 1")
        error_1.append(i)
    # elif os.path.getsize(dir_name) == error_size_2:
    #     print("Pass: error 2")
    #     error_2.append(i)
    elif os.path.getsize(dir_name) == error_size_3:
        print("Pass: error 3")
        error_3.append(i)
    elif os.path.getsize(dir_name) == error_size_4:
        print("Pass: error 4")
        error_4.append(i)
    else :
        valid_struc.append(i)

print("Error_1 have", error_1)
print("Error_3 have", error_3)
print("Error_4 have", error_4)
print("Valid structure have",valid_struc)



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