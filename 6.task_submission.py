# This program is used for cluster only

import time
import subprocess
submit = True
if submit:
    for i in range(120, 200): #0-200 Done
        dir_name = "./output_vasp/NbO_" + str(i)
        command = 'qsub vasp.5.4.4.pbs'
        print("Excuiting No.", i)
        time.sleep(3)
        subprocess.Popen(
             command,
             shell=True,
             cwd=dir_name,
         )
print("All Done")