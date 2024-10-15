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

Nx = dataFile['Nx']
Ny = dataFile['Ny']
Nz = dataFile['Nz']

LID_SPEED = dataFile['LID_SPEED']
VISCOSITY = dataFile['VISCOSITY']
FRANK_CONSTANT = dataFile['FRANK_CONSTANT']
GAMMA = dataFile['GAMMA']
NEMATIC_DENSITY = dataFile['NEMATIC_DENSITY']
ZETA = dataFile['ZETA']

density = dataFile['density']
vel_x = dataFile['vel_x']
vel_y = dataFile['vel_y']
vel_z = dataFile['vel_z']

# Q_xx = dataFile('Q_xx')
# Q_xx = dataFile('Q_xx')
# Q_xx = dataFile('Q_xx')
# Q_xx = dataFile('Q_xx')
# Q_xx = dataFile('Q_xx')

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

       
SUPTITLE = '$N_x$ = %d, $\lambda$ = %.2f, Re = %.3f, Er = %.3f, Ac = %d, \n$Ac_{saloni}$ = %d, U = %.1f, $l_c$ = %.2f$\mu m$,  $t_c$ = %.1fns, t = %.3fs' % (Nx, aspect_ratio, Re, Er, Ac, saloni_Activity, NEMATIC_STRENGTH, nematic_correlation_length * LENGTH_CONVERSION_FACTOR, nematic_correlation_time * TIME_CONVERSION_FACTOR * 1e6)

# Projecting velocities, not averaging
# due to matplotlib array indexing, np.swapaxes method is used

# Mid-plane x-y velocities
vel_x_xy = vel_x[..., int(Nz/2)]
vel_y_xy = vel_y[..., int(Nz/2)]
vel_x_xy = np.swapaxes(vel_x_xy * VELOCITY_CONVERSION_FACTOR, 0, 1)
vel_y_xy = np.swapaxes(vel_y_xy * VELOCITY_CONVERSION_FACTOR, 0, 1)

# Vertical slice of x-z plane
vel_x_xz = vel_x[:,int(Ny/2),:]
vel_z_xz = vel_z[:,int(Ny/2),:]
vel_x_xz = np.swapaxes(vel_x_xz * VELOCITY_CONVERSION_FACTOR, 0, 1)
vel_z_xz = np.swapaxes(vel_z_xz * VELOCITY_CONVERSION_FACTOR, 0, 1)

# Vertical slice of y-z plane
vel_y_yz = vel_y[int(Nx/2)]
vel_z_yz = vel_z[int(Nx/2)]
vel_y_yz = vel_y_yz * VELOCITY_CONVERSION_FACTOR
vel_z_yz = vel_z_yz * VELOCITY_CONVERSION_FACTOR


averaged_speed_xy = np.sqrt(np.square(vel_x_xy) + np.square(vel_y_xy))
averaged_speed_xz = np.sqrt(np.square(vel_x_xz) + np.square(vel_z_xz))
averaged_speed_yz = np.sqrt(np.square(vel_y_yz) + np.square(vel_z_yz))

x_ticks = np.linspace(0,NX_PHYSICAL,5,endpoint=True,dtype=int)
y_ticks = np.linspace(0,NY_PHYSICAL,5,endpoint=True,dtype=int)
z_ticks = np.linspace(0,Nz * LENGTH_CONVERSION_FACTOR,5,endpoint=True,dtype=int)


""" CORRELATION ANALYSIS """
""" Correlation analysis with trimming """
edgeTrim = int(Nx * 0.05)    # number of grid points trimmed off on the edge
drN = 50                     # number of interpolating points
drThresholdLeft = 0
drThresholdRight = int(Nx / 8) * LENGTH_CONVERSION_FACTOR

# edge trimming n points
x, y = np.meshgrid((np.arange(Nx) + 0.5) * LENGTH_CONVERSION_FACTOR, (np.arange(Ny) + 0.5) * LENGTH_CONVERSION_FACTOR) 
[shiftedX, shiftedY] = [x - Nx * LENGTH_CONVERSION_FACTOR / 2,y - Ny * LENGTH_CONVERSION_FACTOR / 2]
length_scale = np.sqrt((Nx ** 2 + Ny ** 2) / 2) * LENGTH_CONVERSION_FACTOR
if edgeTrim != 0:
       shiftedX = shiftedX[edgeTrim:-edgeTrim,edgeTrim:-edgeTrim]
       shiftedY = shiftedY[edgeTrim:-edgeTrim,edgeTrim:-edgeTrim]

       uxTrim = vel_x_xy[edgeTrim:-edgeTrim,edgeTrim:-edgeTrim]
       uyTrim = vel_y_xy[edgeTrim:-edgeTrim,edgeTrim:-edgeTrim]
else:
       shiftedX = shiftedX
       shiftedY = shiftedY

       uxTrim = vel_x_xy
       uyTrim = vel_y_xy

# Applying convolution-correlation theorem
ConvolutionUx = np.fft.ifftn(np.conjugate(np.fft.fftn(uxTrim))*np.fft.fftn(uxTrim))
ConvolutionUy = np.fft.ifftn(np.conjugate(np.fft.fftn(uyTrim))*np.fft.fftn(uyTrim))

Correlation = ConvolutionUx + ConvolutionUy
Correlation = Correlation/Correlation[0,0]
Correlation = np.fft.fftshift(Correlation) # due to FFT algorithm, necessary to shift back


drGrid = np.sqrt(shiftedX**2+shiftedY**2)
mask = np.ones(drGrid.shape)
drList = np.linspace(0,np.max(drGrid[:]),drN)
autocorr = np.zeros(len(drList)-1)
for i in range(len(drList)-1):
       selectedIndices = np.logical_and(np.less_equal(drGrid, mask*drList[i+1]),np.greater_equal(drGrid, mask*drList[i]))
       autocorr[i] = np.mean(Correlation[selectedIndices])

# Fitting
# get rid of NaN values given by log when negative correlation and right side cut-off
dr = drList[:-1]
logCorr = np.log(autocorr)
index = np.isfinite(dr) & np.isfinite(logCorr) & np.greater(dr,drThresholdLeft) & np.less(dr,drThresholdRight)
myfit = np.polyfit(dr[index],logCorr[index],deg=1,cov=True)
fitx = np.linspace(0,NX_PHYSICAL/2,1000)
fity = np.exp(np.poly1d(myfit[0])(fitx))


# Correlation Length Calculation
slope = myfit[0][0]
corrLength = -1/slope
normalizedCorrLength = corrLength/length_scale
variance = myfit[1][0,0]
std = np.sqrt(variance)
error = std * corrLength**2
normalizedError = error/length_scale

FileName = 'Correlation Length.txt'
text1 = 'Correlation Length is %f ' %(normalizedCorrLength)
text2 = 'Reynolds number is %f ' %(Re)
with open(FileName, 'w') as f:
       f.write(text1)
       f.write('\n')
       f.write(text2)


# Make the plot(s)
fig, axes = plt.subplots(1,2, figsize=(12, 5))
# fig.tight_layout(h_pad=2)
fig.suptitle(SUPTITLE)
plt.subplots_adjust(top=0.85)

axes[0].set(xlim=(0,length_scale/3), ylim=(0.01, 1.1),
       xlabel='$\Delta r (\mu m)$', ylabel='C($\Delta$ r)',
       yscale='log',
       title='Auto-Correlation Function (dr)')
axes[0].plot(dr,autocorr,'bo')
axes[0].plot(fitx,fity,'r-',label='$l_c$ = %f' %(normalizedCorrLength))
axes[0].legend()

axes[1].set(xlim=(-length_scale/2, length_scale/2), ylim=(-length_scale/2, length_scale/2),
       xlabel='$\Delta x (\mu m)$', ylabel='$\Delta y (\mu m)$',
       title='Auto-Correlation Function Map')
im=axes[1].contourf(shiftedX, shiftedY, Correlation, 20, cmap='bwr')   
clb = plt.colorbar(im,ax=axes[1])
clb.ax.set_title('$C$')

# plt.show()
plt.savefig('Analysis/Correlation Analysis.pdf')
plt.close()

""" Correlation analysis no trimming """
edgeTrim = 0    # number of grid points trimmed off on the edge
drN = 50                     # number of interpolating points
drThresholdLeft = 0
drThresholdRight = int(Nx / 8) * LENGTH_CONVERSION_FACTOR

# edge trimming n points
x, y = np.meshgrid((np.arange(Nx) + 0.5) * LENGTH_CONVERSION_FACTOR, (np.arange(Ny) + 0.5) * LENGTH_CONVERSION_FACTOR) 
[shiftedX, shiftedY] = [x - Nx * LENGTH_CONVERSION_FACTOR / 2,y - Ny * LENGTH_CONVERSION_FACTOR / 2]
length_scale = np.sqrt((Nx ** 2 + Ny ** 2) / 2) * LENGTH_CONVERSION_FACTOR
uxTrim = vel_x_xy
uyTrim = vel_y_xy

# Applying convolution-correlation theorem
ConvolutionUx = np.fft.ifftn(np.conjugate(np.fft.fftn(uxTrim))*np.fft.fftn(uxTrim))
ConvolutionUy = np.fft.ifftn(np.conjugate(np.fft.fftn(uyTrim))*np.fft.fftn(uyTrim))

Correlation = ConvolutionUx + ConvolutionUy
Correlation = Correlation/Correlation[0,0]
Correlation = np.fft.fftshift(Correlation) # due to FFT algorithm, necessary to shift back


drGrid = np.sqrt(shiftedX**2+shiftedY**2)
mask = np.ones(drGrid.shape)
drList = np.linspace(0,np.max(drGrid[:]),drN)
autocorr = np.zeros(len(drList)-1)
for i in range(len(drList)-1):
       selectedIndices = np.logical_and(np.less_equal(drGrid, mask*drList[i+1]),np.greater_equal(drGrid, mask*drList[i]))
       autocorr[i] = np.mean(Correlation[selectedIndices])

# Fitting
# get rid of NaN values given by log when negative correlation and right side cut-off
dr = drList[:-1]
logCorr = np.log(autocorr)
index = np.isfinite(dr) & np.isfinite(logCorr) & np.greater(dr,drThresholdLeft) & np.less(dr,drThresholdRight)
myfit = np.polyfit(dr[index],logCorr[index],deg=1,cov=True)
fitx = np.linspace(0,length_scale/2,1000)
fity = np.exp(np.poly1d(myfit[0])(fitx))


# Correlation Length Calculation
slope = myfit[0][0]
corrLength = -1/slope
normalizedCorrLength = corrLength/length_scale
variance = myfit[1][0,0]
std = np.sqrt(variance)
error = std * corrLength**2
normalizedError = error/length_scale

FileName = 'Correlation Length no trimming.txt'
text1 = 'Correlation Length is %f ' %(normalizedCorrLength)
text2 = 'Reynolds number is %f ' %(Re)
with open(FileName, 'w') as f:
       f.write(text1)
       f.write('\n')
       f.write(text2)


# Make the plot(s)
fig, axes = plt.subplots(1,2, figsize=(12, 5))
# fig.tight_layout(h_pad=2)
fig.suptitle(SUPTITLE)
plt.subplots_adjust(top=0.85)

axes[0].set(xlim=(0,length_scale/3), ylim=(0.01, 1.1),
       xlabel='$\Delta r (\mu m)$', ylabel='C($\Delta$ r)',
       yscale='log',
       title='Auto-Correlation Function (dr)')
axes[0].plot(dr,autocorr,'bo')
axes[0].plot(fitx,fity,'r-',label='$l_c$ = %f' %(normalizedCorrLength))
axes[0].legend()

axes[1].set(xlim=(-length_scale/2, length_scale/2), ylim=(-length_scale/2, length_scale/2),
       xlabel='$\Delta x (\mu m)$', ylabel='$\Delta y (\mu m)$',
       title='Auto-Correlation Function Map')
im=axes[1].contourf(shiftedX, shiftedY, Correlation, 20, cmap='bwr')   
clb = plt.colorbar(im,ax=axes[1])
clb.ax.set_title('$C$')

# plt.show()
plt.savefig('Analysis/Correlation Analysis no trimming.pdf')
plt.close()

fig, ax = plt.subplots()
fig.suptitle(SUPTITLE)
plt.subplots_adjust(top=0.8)
ax.set(xlim=(0, Nx * LENGTH_CONVERSION_FACTOR), ylim=(0, Ny * LENGTH_CONVERSION_FACTOR),
       xticks=x_ticks, yticks=y_ticks,
       xlabel='X($\mu m$)', ylabel='Y($\mu m$)',
       title='Flow pattern (Z = %d)' %(int(Nz/2)))
x, y = np.meshgrid((np.arange(Nx) + 0.5) * LENGTH_CONVERSION_FACTOR, (np.arange(Ny) + 0.5) * LENGTH_CONVERSION_FACTOR)
im = ax.contourf(x, y, averaged_speed_xy, 80, cmap='RdYlBu')
clb = plt.colorbar(im,ax=ax)
clb.ax.set_title('u($\mu$m/s)')
ax.streamplot(x, y, vel_x_xy, vel_y_xy, density=1.5, color='black')

plt.savefig('frames/flow_t%05d.png' %(Frame / DISK_TIME))
plt.close()

fig, ax = plt.subplots()
fig.suptitle(SUPTITLE)
plt.subplots_adjust(top=0.8)
ax.set(xlim=(0, Nx * LENGTH_CONVERSION_FACTOR), ylim=(0, Ny * LENGTH_CONVERSION_FACTOR),
       xticks=x_ticks, yticks=y_ticks,
       xlabel='X($\mu m$)', ylabel='Y($\mu m$)',
       title='Flow pattern (Z = %d)' %(int(Nz/2)))
x, y = np.meshgrid((np.arange(Nx) + 0.5) * LENGTH_CONVERSION_FACTOR, (np.arange(Ny) + 0.5) * LENGTH_CONVERSION_FACTOR)
im = ax.contourf(x, y, averaged_speed_xy, 80, cmap='RdYlBu')
clb = plt.colorbar(im,ax=ax)
clb.ax.set_title('u($\mu$m/s)')
ax.quiver(x[::2, ::2], y[::2, ::2], vel_x_xy[::2, ::2], vel_y_xy[::2, ::2],
              pivot='mid', units='inches')
plt.savefig('frames/flowfield_t%05d.png' %(Frame / DISK_TIME))
plt.close()


