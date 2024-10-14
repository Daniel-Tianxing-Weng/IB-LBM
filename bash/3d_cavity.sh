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

WORKING_DIR="../Data/${11}/Nx${1}Ny${2}Nz${3}t_num${4}u_lid${5}Gamma${6}L${7}U${8}A${9}zeta${10}"
mkdir -p $WORKING_DIR

# echo $WORKING_DIR
# echo ${12}

g++ -std=c++17 -o $WORKING_DIR/3DLBM src/LBM_3D_cavity.cpp
cd $WORKING_DIR
./3DLBM $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}

cd ../../../IB-LBM
bash bash/analysis.sh $WORKING_DIR
# sbatch analysis.sh $WORKING_DIR
