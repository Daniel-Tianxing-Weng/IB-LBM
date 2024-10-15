# Importing required module
import subprocess
import numpy as np
import os

# Using system() method to
# execute shell command

DIR = 'test'

for N in [20]:

    for j in range(1):

        # if j == 0:
        #     continue

        for i in range(2):

            if i == 0:
                continue

            SUBSET = 'set%d' %(i)
            if not os.path.exists(os.path.join(os.getcwd(), 'temp/', SUBSET)):
                os.mkdir(os.path.join(os.getcwd(), 'temp/', SUBSET))

            Re = i/100

            # System Geometry
            Nx = N + 1
            Ny = N + 1
            Nz = N + 1

            # Simulation timescale
            t_num = 6000
            t_disk = 1000
            t_lbm = 0.9

            # Controlled parameters
            vel_back =  3 * Re / ((t_lbm - 0.5) * N)
            # print(vel_back)
            Gamma = 100
            L = 3e-6
            U = 0.7
            A = 1e-3 / np.square(N/60)
            zeta = (0.073)/ np.square(N/60)
            xi = 1

            Parameters = str(Nx)+' '+str(Ny)+' '+str(Nz)+' '+str(t_num)+' '+str(t_disk)+' '+str(t_lbm)+' '+str(vel_back)+' '+str(Gamma)+' '+str(L)+' '+str(U)+' '+str(A)+' '+str(zeta)+' '+str(xi)

            FileName = "SimulationParameters.txt"
            FileID = os.path.join(os.getcwd(), 'temp/', SUBSET, FileName)
                
            with open(FileID, "w") as f:
                f.write("Nx Ny Nz TOTAL_TIME DISK_TIME LBM_TIME LID_SPEED GAMMA FRANK_CONSTANT NEMATIC_STRENGTH NEMATIC_DENSITY ZETA FLOW_ALIGNING\n")
                f.write(Parameters)

            # Process = subprocess.Popen(
            #     [       
            #         'bash',
            #         'bash/3d_cavity.sh',
            #         DIR,
            #         SUBSET,
            #     ],
            # )

            Process = subprocess.Popen(
                [       
                    'sbatch',
                    'bash/3d_cavity.sh',
                    DIR,
                    SUBSET,
                ],
            )

        Process.wait()