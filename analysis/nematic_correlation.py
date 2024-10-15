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

import os

import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS #

# READ-IN PARAMETER FILE
dataFile = np.load('Data.npz')

Nx = dataFile('Nx')
Ny = dataFile('Ny')
Nz = dataFile('Nz')

LID_SPEED = dataFile('LID_SPEED')
VISCOSITY = dataFile('VISCOSITY')
FRANK_CONSTANT = dataFile('FRANK_CONSTANT')
GAMMA = dataFile('GAMMA')
NEMATIC_DENSITY = dataFile('NEMATIC_DENSITY')
ZETA = dataFile('ZETA')

density = dataFile('density')
vel_x = dataFile('vel_x')
vel_y = dataFile('vel_y')
vel_z = dataFile('vel_z')

Q_xx = dataFile('Q_xx')
Q_yy = dataFile('Q_yy')
Q_xy = dataFile('Q_xy')
Q_yz = dataFile('Q_yz')
Q_xz = dataFile('Q_xz')

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

       
SUPTITLE = '$N_x$ = %d, $\lambda$ = %.2f, Re = %.3f, Er = %.3f, Ac = %d, \n$Ac_{saloni}$ = %d, U = %.1f, $l_c$ = %.2f$\mu m$,  $t_c$ = %.1fs, t = %.3fs' % (Nx, aspect_ratio, Re, Er, Ac, saloni_Activity, NEMATIC_STRENGTH, nematic_correlation_length * LENGTH_CONVERSION_FACTOR, nematic_correlation_time * TIME_CONVERSION_FACTOR, Frame * TIME_CONVERSION_FACTOR)



# Projecting velocities, not averaging

x_ticks = np.linspace(0,NX_PHYSICAL,5,endpoint=True,dtype=int)
y_ticks = np.linspace(0,NY_PHYSICAL,5,endpoint=True,dtype=int)
z_ticks = np.linspace(0,Nz * LENGTH_CONVERSION_FACTOR,5,endpoint=True,dtype=int)


""" NEMATIC ANALYSIS """
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
       

""" CORRELATION ANALYSIS """
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