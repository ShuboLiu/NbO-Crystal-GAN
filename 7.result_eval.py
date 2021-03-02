import os
import sys
import numpy as np 
import shutil
import subprocess
import re

np.set_printoptions(threshold=np.inf)

data = np.load(r"./loss_image/fake_imgs_after_post_process.npy")
print("Shape of import data is", data.shape)

"""
error size 1 [0]: internal error in WFINIT: orbitals linearily dependent at random-number initialization
error size 2 [13]: Unreasonable structure, VASP will ignore
error size 3 [1]: scaLAPACK: Routine ZPOTRF ZTRTRI failed! LAPACK: Routine ZTRTRI failed!
    尝试修改晶格常数（一般扩大）;
    尝试修改INCAR中的ALGO到VERY_FAST
error size 4 [6]: REAL_OPT: internal ERROR:           0           2          -1           0
    据说要改变ENCUT可解决，实测不可以，太大没影响；太小变成error 1
error size 5 [9]: KILLED BY SIGNAL: 9 (Killed)
    伴随“PZSTEIN parameter number    X had an illegal value” 错误
    参见：http://www.error.wiki/Exit_status_of_rank_9:_killed_by_signal_9
    尝试添加 LHFCALC = .TRUE.
error size 6 [5]: ERROR in subspace rotation PSSYEVX: not enough eigenvalues found
    参见: http://www.error.wiki/ERROR_in_subspace_rotation_PSSYEVX:_not_enough_eigenvalues_found
    尝试将AlGO改为Normal
"""

def TrFal(input_num):
    if input_num < 0:
        return False
    else:
        return True

error_1 = []; error_2 = []; error_3 = []; 
error_4 = []; error_5 = []; error_6 = []; 
valid_struc = []
Total = list(range(len(data)))
for i in range(len(data)):
    dir_name = "./output_vasp/NbO_" + str(i)
    out_name = os.path.join(dir_name, "out")
    vasp_output = open(out_name, "r").read()
    if TrFal(vasp_output.find("internal error in WFINIT: orbitals linearily dependent at random-number initialization")):
        error_1.append(i)
    if TrFal(vasp_output.find("POSCAR, INCAR and KPOINTS ok, starting setup")): pass
    else:
        error_2.append(i) # error_2 structures should be skipped
    if TrFal(vasp_output.find("scaLAPACK: Routine ZPOTRF ZTRTRI failed!")) or TrFal(vasp_output.find("LAPACK: Routine ZPOTRF failed!")):
        error_3.append(i)
    if TrFal(vasp_output.find("REAL_OPT: internal ERROR:")):
        error_4.append(i)
    if TrFal(vasp_output.find("KILLED BY SIGNAL: 9 (Killed)")): 
        error_5.append(i)
    if TrFal(vasp_output.find("ERROR in subspace rotation PSSYEVX: not enough eigenvalues found")): 
        error_6.append(i)

    if TrFal(vasp_output.find("reached required accuracy - stopping structural energy minimisation")): 
        valid_struc.append(i)

Count = list(set(error_1) | set(error_2) |set(error_3) | set(error_4) | set(error_5) | set(error_6)| set(valid_struc))
other = list(set(Total).difference(set(Count))) 
print("Error_1 have", error_1)
print("Error_2 have", error_2)
print("Error_3 have", error_3)
print("Error_4 have", error_4)
print("Error_5 have", error_5)
print("Error_6 have", error_6)
print("Error_Other have", other)
print("Valid structure have", valid_struc)

print("\n  *** According to the algorithm, you may double check 'other' ***  \n")

# Auto Debug
auto_debug_mode = False
if auto_debug_mode :
    for i in error_5:
        dir_name = "./output_vasp/NbO_" + str(i)
        out_name = os.path.join(dir_name, "out")
        INCAR_dir = os.path.join(dir_name, "INCAR")
        shutil.copyfile("./POSCAR/TroubleShooting/error_5/INCAR", INCAR_dir)
    for i in error_6:
        dir_name = "./output_vasp/NbO_" + str(i)
        out_name = os.path.join(dir_name, "out")
        INCAR_dir = os.path.join(dir_name, "INCAR")
        shutil.copyfile("./POSCAR/TroubleShooting/error_6/INCAR", INCAR_dir)

# Rerun VASP calculation

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