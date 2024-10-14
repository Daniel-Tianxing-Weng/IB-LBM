#!/bin/bash
#SBATCH -J lbm
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --mem 10G
#SBATCH -p short
#SBATCH -t 06:00:00
#SBATCH --output=lbm_%j.out
#SBATCH --error=lbm_%j.err

# $1=Nx 
# $2=Ny
# $3=Nz
# $4=t_num
# $5=vel_back
# $6=Gamma
# $7=L
# $8=U
# $9=A
# ${10}=-zeta

WORKING_DIR="Data/${11}/Nx${1}Ny${2}Nz${3}t_num${4}u_lid${5}Gamma${6}L${7}U${8}A${9}zeta${10}"
mkdir -p $WORKING_DIR

# echo $WORKING_DIR
# echo ${12}

g++ -std=c++17 -o $WORKING_DIR/3DLBM src/LBM_3D_cavity.cpp
cd $WORKING_DIR
./3DLBM $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}

scp ../../${12}/analysis_packages/analysis_correlation.py analysis_correlation.py
scp ../../${12}/analysis_packages/analysis_nematic.py analysis_nematic.py
scp ../../${12}/analysis_packages/Organize_data.m Organize_data.m
scp ../../${12}/analysis_packages/Plotting.m Plotting.m

source ../../${12}/.venv/bin/activate

python3 -Wi analysis_correlation.py
python3 -Wi analysis_nematic.py


mkdir -p movies
module load ffmpeg
ffmpeg -r 24 -i frames/flow_t%05d.png -vcodec mpeg4 -y movies/flow.mp4
ffmpeg -r 24 -i frames/orientation_t%05d.png -vcodec mpeg4 -y movies/orientation.mp4

cd ../../${12}
