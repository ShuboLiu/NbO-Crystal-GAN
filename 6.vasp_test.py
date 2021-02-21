import os
import sys
import numpy as np 
import shutil
import subprocess
import time

shutil.rmtree("output_vasp")
os.makedirs("output_vasp", exist_ok = True)

data = np.load(r"./loss_image/fake_imgs_after_post_process.npy")

for i in range(len(data)):
    dir_name = "./output_vasp/NbO_" + str(i)
    os.makedirs(dir_name)

    # 创建POSCAR
    POSCAR_dir = os.path.join(dir_name, "POSCAR")
    with open(POSCAR_dir, 'w') as f:
        f.write('Nb O \n') 
        f.write('1.0 \n') 
        for j in range(0, 3):
            f.write("%f %f %f\n" 
                % (data[i, j, 0], data[i, j, 1], data[i, j, 2]))
        f.write('Nb O \n')
        f.write('1 1 \n')
        f.write('Direct \n')
        for j in range(3, 5):
            f.write("%f %f %f\n" 
                % (data[i, j, 0], data[i, j, 1], data[i, j, 2]))
    

    # 复制INCAR
    INCAR_dir = os.path.join(dir_name, "INCAR")
    shutil.copyfile("./POSCAR/INCAR", INCAR_dir)

    # 复制POTCAR
    POTCAR_dir = os.path.join(dir_name, "POTCAR")
    shutil.copyfile("./POSCAR/POTCAR", POTCAR_dir)

    # 复制KPOINTS
    KPOINTS_dir = os.path.join(dir_name, "KPOINTS")
    shutil.copyfile("./POSCAR/KPOINTS", KPOINTS_dir)

print("All file set")

def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)

def TrFal(input_num):
    if input_num < 0:
        return False
    else:
        return True

for i in range(len(data)): 
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
        p.wait(120) #只设定120S wall time
    except subprocess.TimeoutExpired:
        p.kill()
    
fail_struc = []
for i in range(len(data)): 
    dir_name = "./output_vasp/NbO_" + str(i)
    out_name = os.path.join(dir_name, "out")
    vasp_output = open(out_name, "r").read()
    if TrFal(vasp_output.find("POSCAR, INCAR and KPOINTS ok, starting setup")): pass
    else:
        fail_struc.append(i)

print("Obtain %.2f%% Chemical Effective Structure " % (1 - len(fail_struc)/len(data)) * 100)
print("All Done")