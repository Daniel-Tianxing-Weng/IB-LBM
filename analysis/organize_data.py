import pandas as pd
import numpy as np
from glob import glob
import os

Parameters = './SimulationParameters.txt'
pf = pd.read_csv(Parameters, delimiter='\s', header=0, engine='python')
Nx = int(pf['Nx'].to_numpy()[0])
Ny = int(pf['Ny'].to_numpy()[0])
Nz = int(pf['Nz'].to_numpy()[0])
LID_SPEED = pf['LID_SPEED'].to_numpy()[0]

t_lbm = pf['LBM_TIME'].to_numpy()[0]
VISCOSITY = (t_lbm - 0.5) / 3

GAMMA = pf['GAMMA'].to_numpy()[0]
ZETA = pf['ZETA'].to_numpy()[0]
NEMATIC_DENSITY = pf['NEMATIC_DENSITY'].to_numpy()[0]
NEMATIC_STRENGTH = pf['NEMATIC_STRENGTH'].to_numpy()[0]
FRANK_CONSTANT = pf['FRANK_CONSTANT'].to_numpy()[0]

TOTAL_TIME = pf['TOTAL_TIME'].to_numpy()[0]
DISK_TIME = pf['DISK_TIME'].to_numpy()[0]

TOTAL_TIME = DISK_TIME * len(glob(os.getcwd()+'/data/*.dat'))
time_steps = int(TOTAL_TIME / DISK_TIME)
shape = (Ny, Nx, Nz)
temporal_shape = (time_steps, Ny, Nx, Nz)

density = np.zeros(temporal_shape)
vel_x = np.zeros(temporal_shape)
vel_y = np.zeros(temporal_shape)
vel_z = np.zeros(temporal_shape)

Q_xx = np.zeros(temporal_shape)
Q_yy = np.zeros(temporal_shape)
Q_xy = np.zeros(temporal_shape)
Q_yz = np.zeros(temporal_shape)
Q_xz = np.zeros(temporal_shape)

scalar_order_parameter = np.zeros(temporal_shape)
director_x = np.zeros(temporal_shape)
director_y = np.zeros(temporal_shape)
director_z = np.zeros(temporal_shape)

density_Temp = np.zeros(shape)
vel_x_Temp = np.zeros(shape)
vel_y_Temp = np.zeros(shape)
vel_z_Temp = np.zeros(shape)
Q_xx_Temp = np.zeros(shape)
Q_yy_Temp = np.zeros(shape)
Q_xy_Temp = np.zeros(shape)
Q_yz_Temp = np.zeros(shape)
Q_xz_Temp = np.zeros(shape)

for t in range(time_steps):
    data_file = 'data/fluid_t%d.dat' %((t+1) * DISK_TIME)
    df = pd.read_csv(data_file, delimiter='\s', header=0, engine='python') 

    # read in with delimiter of one-space separation
    # convert input into numpy array
    density_Temp = np.reshape(df['density'].to_numpy(), shape)
    vel_x_Temp = np.reshape(df['vel_x'].to_numpy(), shape)
    vel_y_Temp = np.reshape(df['vel_y'].to_numpy(), shape)
    vel_z_Temp = np.reshape(df['vel_z'].to_numpy(), shape)
    Q_xx_Temp = np.reshape(df['Q_xx'].to_numpy(), shape)
    Q_yy_Temp = np.reshape(df['Q_yy'].to_numpy(), shape)
    Q_xy_Temp = np.reshape(df['Q_xy'].to_numpy(), shape)
    Q_yz_Temp = np.reshape(df['Q_yz'].to_numpy(), shape)
    Q_xz_Temp = np.reshape(df['Q_xz'].to_numpy(), shape)

    density[t] = density_Temp
    vel_x[t] = vel_x_Temp
    vel_y[t] = vel_y_Temp
    vel_z[t] = vel_z_Temp
    Q_xx[t] = Q_xx_Temp
    Q_yy[t] = Q_yy_Temp
    Q_xy[t] = Q_xy_Temp
    Q_yz[t] = Q_yz_Temp
    Q_xz[t] = Q_xz_Temp

    for X in range(Nx):
       for Y in range(Ny):
              for Z in range(Nz):
                     Q_total = np.array([[Q_xx_Temp[X,Y,Z], Q_xy_Temp[X,Y,Z], Q_xz_Temp[X,Y,Z]],
                                          [Q_xy_Temp[X,Y,Z], Q_yy_Temp[X,Y,Z], Q_yz_Temp[X,Y,Z]],
                                          [Q_xz_Temp[X,Y,Z], Q_yz_Temp[X,Y,Z], -Q_xx_Temp[X,Y,Z]-Q_yy_Temp[X,Y,Z]]])
                     eigs = np.linalg.eig(Q_total)

                     scalar_order_parameter[t,X,Y,Z] = np.max(eigs[0])    # maximum eigenvalue
                     director_x[t,X,Y,Z] = eigs[1][0,np.argmax(eigs[0])]  # nx
                     director_y[t,X,Y,Z] = eigs[1][1,np.argmax(eigs[0])]  # ny
                     director_z[t,X,Y,Z] = eigs[1][2,np.argmax(eigs[0])]  # nz

    
    

np.savez_compressed('Data.npz', 
                    Nx=Nx, Ny=Ny, Nz=Nz, 
                    VISCOSITY=VISCOSITY, GAMMA=GAMMA, 
                    LID_SPEED=LID_SPEED, ZETA=ZETA,
                    FRANK_CONSTANT=FRANK_CONSTANT, 
                    NEMATIC_DENSITY=NEMATIC_DENSITY,
                    
                    density=density,
                    vel_x=vel_x, vel_y=vel_y, vel_z=vel_z,
                    S=scalar_order_parameter, nx=director_x, ny=director_y, nz=director_z
                    )
