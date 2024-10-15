# Importing required module
import subprocess
import numpy as np

# Using system() method to
# execute shell command

for N in [60]:
    for j in range(1):

        # if j == 0:
        #     continue

        for i in range(2):
            
            if i == 0:
                continue
            Re = i/100
            # print(Re)

            Nx = N + 1
            Ny = N + 1
            Nz = int(N / 6) + 1
            t_num = 60000

            vel_back = np.round(2/15 * Re / N, decimals=7)
            # print(vel_back)
            Gamma = 100
            L = 3e-6
            U = 0.7
            A = np.round(1e-3 / np.square(N/60), decimals=5)
            zeta = np.round((0.073)/ np.square(N/60), decimals=4)

            Process = subprocess.Popen(
                [       
                    'bash',
                    'batch_scripts/myjob_active.sh',
                    str(Nx),
                    str(Ny),
                    str(Nz),
                    str(t_num),
                    str(vel_back),
                    str(Gamma),
                    str(L),
                    str(U),
                    str(A),
                    str(zeta),
                    'active/longer_time',
                    '../..',
                ],
            )

        Process.wait()