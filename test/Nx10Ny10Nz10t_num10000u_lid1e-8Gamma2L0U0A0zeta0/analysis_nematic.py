""" This script aims at analyzing the data output by the c++ script of 
lattice boltzmann simulations, and consists of three main parts:

1. Read-in data using pandas DataFrame object,
    - pass simulation parameters into analysis script
    - grab all files in the data directory
2. plot the flow field and director field at each given time-frame
    - projecting z-axis flow onto xy-plane (checkmark)
    - 3d quiver plot for directors
3. perform correlation length analysis for the velocity data
    - all time series data
    - averaging over z-axis

A further goal is to incorporate this script into an automated workflow
on the high-performance cluster. (incorporate a Template directory on cluster
and only modify parameters via shell commands)

 """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob


# PARAMETERS #
""" TODO: 
1. in the cpp script output simulation parameters in the initiation();
2. in the python script read in the output, use it as analysis parameters

as a note: the required parameters for outputs are:
- system geometry: Nx, Ny, Nz
- dynamical variable: viscosity, lid-speed, total time
- hybrid parameters: relaxation, Frank constant, nematic phase indicator
- dimensionless numbers: Re, Er, Ac, aspect ratio (Brinkman term)
"""

# READ-IN PARAMETER FILE
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

# OUTPUT A NOTE WITH DIMENSIONLESS NUMBERS
Re = (Nx - 1) * LID_SPEED / VISCOSITY
Er = (Nx - 1) * LID_SPEED / (FRANK_CONSTANT * GAMMA)
Ac = -ZETA * (Nx - 1) / (VISCOSITY * LID_SPEED)
aspect_ratio = (Nz - 1) / (Nx - 1)
nematic_correlation_length = np.sqrt(FRANK_CONSTANT / NEMATIC_DENSITY)
active_correlation_length = np.sqrt(FRANK_CONSTANT / -ZETA)
nematic_correlation_time = 1 / (NEMATIC_DENSITY * GAMMA)
saloni_Activity = (Nx - 1) * np.sqrt(-ZETA / FRANK_CONSTANT)

# CALCULATE CONVERSION FACTORS
NX_PHYSICAL = 3000                               # in um
NY_PHYSICAL = 3000 
NZ_PHYSICAL = int(NX_PHYSICAL * aspect_ratio)
VISCOSITY_PHYSICAL = 4e-3 / 1.05e3 * 1e12        # in um2/s
LENGTH_CONVERSION_FACTOR = NX_PHYSICAL / (Nx - 1)      # in um
TIME_CONVERSION_FACTOR = LENGTH_CONVERSION_FACTOR**2 * VISCOSITY / VISCOSITY_PHYSICAL # in s
VELOCITY_CONVERSION_FACTOR = LENGTH_CONVERSION_FACTOR / TIME_CONVERSION_FACTOR # in um/s
LID_SPEED_PHYSICAL = LID_SPEED * VELOCITY_CONVERSION_FACTOR   # in um/s

DENSITY_PHYSICAL = 1.05e3 # in kg m-3

# MAKE ANALYSIS DIRECTORY #
if not os.path.exists('frames'):
       os.mkdir('frames')
if not os.path.exists('slices'):
       os.mkdir('slices')
if not os.path.exists('Analysis'):
       os.mkdir('Analysis')

# READ-IN DATA FILE #
TOTAL_TIME = DISK_TIME * len(glob(os.getcwd()+'/data/*.dat'))
Frames = range(DISK_TIME,TOTAL_TIME+DISK_TIME,DISK_TIME)
for Frame in Frames:
       
       SUPTITLE = '$N_x$ = %d, $\lambda$ = %.2f, Re = %.3f, Er = %.3f, Ac = %d, \n$Ac_{saloni}$ = %d, U = %.1f, $l_c$ = %.2f$\mu m$,  $t_c$ = %.1fs, t = %.3fs' % (Nx, aspect_ratio, Re, Er, Ac, saloni_Activity, NEMATIC_STRENGTH, nematic_correlation_length * LENGTH_CONVERSION_FACTOR, nematic_correlation_time * TIME_CONVERSION_FACTOR, Frame * TIME_CONVERSION_FACTOR)

       data_file = 'data/fluid_t%d.dat' %(Frame)
       df = pd.read_csv(data_file, delimiter='\s', header=0, engine='python') 

       # read in with delimiter of one-space separation
       # convert input into numpy array
       shape = (Nx, Ny, Nz)
       density = np.reshape(df['density'].to_numpy(), shape)
       vel_x = np.reshape(df['vel_x'].to_numpy(), shape)
       vel_y = np.reshape(df['vel_y'].to_numpy(), shape)
       vel_z = np.reshape(df['vel_z'].to_numpy(), shape)
       Q_xx = np.reshape(df['Q_xx'].to_numpy(), shape)
       Q_yy = np.reshape(df['Q_yy'].to_numpy(), shape)
       Q_xy = np.reshape(df['Q_xy'].to_numpy(), shape)
       Q_yz = np.reshape(df['Q_yz'].to_numpy(), shape)
       Q_xz = np.reshape(df['Q_xz'].to_numpy(), shape)

       # Projecting velocities, not averaging

       x_ticks = np.linspace(0,NX_PHYSICAL,5,endpoint=True,dtype=int)
       y_ticks = np.linspace(0,NY_PHYSICAL,5,endpoint=True,dtype=int)
       z_ticks = np.linspace(0,Nz * LENGTH_CONVERSION_FACTOR,5,endpoint=True,dtype=int)


       """ NEMATIC ANALYSIS """

       Q_xx = np.reshape(df['Q_xx'].to_numpy(), shape)
       Q_yy = np.reshape(df['Q_yy'].to_numpy(), shape)
       Q_xy = np.reshape(df['Q_xy'].to_numpy(), shape)
       Q_yz = np.reshape(df['Q_yz'].to_numpy(), shape)
       Q_xz = np.reshape(df['Q_xz'].to_numpy(), shape)

       scalar_order_parameter = np.zeros(shape)
       director_x = np.zeros(shape)
       director_y = np.zeros(shape)
       director_z = np.zeros(shape)
       for X in range(Nx):
              for Y in range(Ny):
                     for Z in range(Nz):
                            Q_total = np.array([[Q_xx[X,Y,Z], Q_xy[X,Y,Z], Q_xz[X,Y,Z]],
                                                 [Q_xy[X,Y,Z], Q_yy[X,Y,Z], Q_yz[X,Y,Z]],
                                                 [Q_xz[X,Y,Z], Q_yz[X,Y,Z], -Q_xx[X,Y,Z]-Q_yy[X,Y,Z]]])
                            eigs = np.linalg.eig(Q_total)

                            scalar_order_parameter[X,Y,Z] = np.max(eigs[0])    # maximum eigenvalue
                            director_x[X,Y,Z] = eigs[1][0,np.argmax(eigs[0])]  # nx
                            director_y[X,Y,Z] = eigs[1][1,np.argmax(eigs[0])]  # ny
                            director_z[X,Y,Z] = eigs[1][2,np.argmax(eigs[0])]  # nz


       projected_nx = director_x[:,:,int(Nz/2)] / np.sqrt(1 - np.square(director_z[:,:,int(Nz/2)]))
       projected_ny = director_y[:,:,int(Nz/2)] / np.sqrt(1 - np.square(director_z[:,:,int(Nz/2)]))

       fig, ax = plt.subplots()
       fig.suptitle(SUPTITLE)
       plt.subplots_adjust(top=0.8)
       ax.set(xlim=(0, Nx * LENGTH_CONVERSION_FACTOR), ylim=(0, Ny * LENGTH_CONVERSION_FACTOR),
              xticks=x_ticks, yticks=y_ticks,
              xlabel='X($\mu m$)', ylabel='Y($\mu m$)',
              title='Orientation Configuration (Z = %d)' %(int(Nz/2)))
       x, y = np.meshgrid((np.arange(Nx) + 0.5) * LENGTH_CONVERSION_FACTOR, (np.arange(Ny) + 0.5) * LENGTH_CONVERSION_FACTOR)
       im = ax.contourf(x, y, np.swapaxes(scalar_order_parameter[:,:,int(Nz/2)], 0, 1), 20, cmap='bwr')
       clb = plt.colorbar(im,ax=ax)
       clb.ax.set_title('S')
       ax.quiver(x[::2, ::2], y[::2, ::2], projected_nx[::2, ::2], projected_ny[::2, ::2], units='xy', headlength=0, headaxislength=0, color='black', pivot='mid')


       plt.savefig('frames/orientation_t%05d.png' %(Frame / DISK_TIME))
       plt.close()
              
       # # 3D projections of orientation fields
       # fig, axes = plt.subplots(2,2, figsize=(12,9))
       # # fig.tight_layout(h_pad=2)
       # fig.suptitle('3D Q-field for ' + SUPTITLE)
       # plt.subplots_adjust(top=0.92)

       # axes[0,0].set(xlim=(0, Nx), ylim=(0, Ny),
       #        xticks=x_ticks, yticks=y_ticks,
       #        xlabel='X', ylabel='Y',
       #        title='Orientation Configuration (Z = %d)' %(int(Nz/2)))
       # x, y = np.meshgrid(np.arange(Nx) + 0.5, np.arange(Ny) + 0.5)
       # im1 = axes[0,0].contourf(x, y, np.swapaxes(scalar_order_parameter[:,:,int(Nz/2)], 0, 1), 20, cmap='bwr')
       # clb = plt.colorbar(im1,ax=axes[0,0])
       # clb.ax.set_title('S')
       # axes[0,0].quiver(x, y, np.swapaxes(director_x[:,:,int(Nz/2)], 0, 1), np.swapaxes(director_y[:,:,int(Nz/2)], 0, 1), units='xy', headlength=0, headaxislength=0, color='black', pivot='mid')

       # axes[0,1].set(xlim=(0, Nz), ylim=(0, Ny),
       #        xticks=z_ticks, yticks=y_ticks,
       #        xlabel='Z', ylabel='Y',
       # #    aspect=0.5,
       #        title='Orientation Configuration (X = %d)' %(int(Nx/2)))
       # z, y = np.meshgrid(np.arange(Nz) + 0.5, np.arange(Ny) + 0.5)
       # im2 = axes[0,1].contourf(z, y, scalar_order_parameter[int(Nx/2)], 20, cmap='bwr')
       # clb = plt.colorbar(im2,ax=axes[0,1])
       # clb.ax.set_title('S')
       # axes[0,1].quiver(z, y, director_z[int(Nx/2)], director_y[int(Nx/2)], units='xy', headlength=0, headaxislength=0, color='black', pivot='mid')

       # axes[1,0].set(xlim=(0, Nx), ylim=(0, Nz),
       #        xticks=x_ticks, yticks=z_ticks,
       #        xlabel='X', ylabel='Z',
       # #    aspect=2,
       #        title='Orientation Configuration (Y = %d)' %(int(Ny/2)))
       # x, z = np.meshgrid(np.arange(Nx) + 0.5, np.arange(Nz) + 0.5)
       # im3 = axes[1,0].contourf(x, z, np.swapaxes(scalar_order_parameter[:,int(Ny/2),:], 0, 1), 20, cmap='bwr')
       # clb = plt.colorbar(im3,ax=axes[1,0])
       # clb.ax.set_title('S')
       # axes[1,0].quiver(x, z, np.swapaxes(director_x[:,int(Ny/2),:], 0, 1), np.swapaxes(director_z[:,int(Ny/2),:], 0, 1), units='xy', headlength=0, headaxislength=0, color='black', pivot='mid')
       # plt.savefig('slices/Q Projection t = %d.pdf' %(Frame))
       # plt.close()

       """ CORRELATION ANALYSIS """
       if Frame == TOTAL_TIME:
              # Projecting onto the mid-plane
              # and only take the middle section
              # S = np.mean(projected_nx ** 2, axis=0) 
              edgeTrim_x = int(Nx * 0.3) 
              edgeTrim_y = int(Ny * 0.4) 
              S = 2 * (np.mean(projected_ny[edgeTrim_x:-edgeTrim_x, -edgeTrim_y:] ** 2, axis=0) - 1/2)
              """ For some weird reason, nx gives negative values while ny gives positive """
              max_range = (edgeTrim_y) * LENGTH_CONVERSION_FACTOR
              dy = ((edgeTrim_y) - np.arange(edgeTrim_y)) * LENGTH_CONVERSION_FACTOR

              # Make the plot(s)
              fig, ax = plt.subplots()
              fig.suptitle(SUPTITLE)
              plt.subplots_adjust(top=0.8)

              ax.set(
                     xlim=(0, max_range), ylim=(0, 1.1),
                     xlabel='$\Delta y (\mu m)$', ylabel='S($\Delta$y)',
                     title='Scalar Order Parameter ($\Delta y$)')
              ax.plot(dy,S,'bo')
              # axes[0].plot(fitx,fity,'r-',label='$l_c$ = %f' %(normalizedCorrLength))
              # axes[0].legend()
              plt.savefig('Analysis/Scalar Order Parameter Analysis with Eigenvectors.pdf')
              plt.close()

              # Use the scalar order parameter on its own
              S = 2 * np.mean(scalar_order_parameter[:,:,int(Nz/2)], axis=0)
              dy = (Ny - np.arange(Ny)) * LENGTH_CONVERSION_FACTOR

              # Make the plot(s)
              fig, ax = plt.subplots()
              fig.suptitle(SUPTITLE)
              plt.subplots_adjust(top=0.8)

              ax.set(xlim=(0,Ny * LENGTH_CONVERSION_FACTOR), ylim=(0, 1.1),
                     xlabel='$\Delta y (\mu m)$', ylabel='S($\Delta$y)',
                     title='Scalar Order Parameter ($\Delta y$)')
              ax.plot(dy,S,'bo')
              # axes[0].plot(fitx,fity,'r-',label='$l_c$ = %f' %(normalizedCorrLength))
              # axes[0].legend()
              plt.savefig('Analysis/Scalar Order Parameter Analysis with Eigenvalues.pdf')
              plt.close()