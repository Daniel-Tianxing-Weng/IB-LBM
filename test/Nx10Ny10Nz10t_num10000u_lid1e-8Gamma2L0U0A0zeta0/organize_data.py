import pandas as pd
import numpy as np
from glob import glob
import os

Parameters = './AnalysisParameters.dat'
pf = pd.read_csv(Parameters, delimiter='\s', header=0, engine='python')
Nx = pf['Nx'].to_numpy()[0]
Ny = pf['Ny'].to_numpy()[0]
Nz = pf['Nz'].to_numpy()[0]
LID_SPEED = pf['LID_SPEED'].to_numpy()[0]
VISCOSITY = pf['VISCOSITY'].to_numpy()[0]
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

density_Temp = np.zeros(shape)
vel_x_Temp = np.zeros(shape)
vel_y_Temp = np.zeros(shape)
vel_z_Temp = np.zeros(shape)
Q_xx_Temp = np.zeros(shape)
Q_yy_Temp = np.zeros(shape)
Q_xy_Temp = np.zeros(shape)
Q_yz_Temp = np.zeros(shape)
Q_xz_Temp = np.zeros(shape)

for t in range(1,time_steps+1):
    data_file = 'data/fluid_t%d.dat' %(t * DISK_TIME)
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

    density[t-1] = density_Temp
    vel_x[t-1] = vel_x_Temp
    vel_y[t-1] = vel_y_Temp
    vel_z[t-1] = vel_z_Temp
    Q_xx[t-1] = Q_xx_Temp
    Q_yy[t-1] = Q_yy_Temp
    Q_xy[t-1] = Q_xy_Temp
    Q_yz[t-1] = Q_yz_Temp
    Q_xz[t-1] = Q_xz_Temp

np.savez_compressed('Data.npz')
