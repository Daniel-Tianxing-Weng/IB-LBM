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

WORKING_DIR="../Data/${1}/${2}"
mkdir -p $WORKING_DIR

mv temp/${2}/SimulationParameters.txt $WORKING_DIR/SimulationParameters.txt

g++ -std=c++17 -o $WORKING_DIR/2DIBLBM ./src/IBLBM_2D_Poiseuille.cpp
cd $WORKING_DIR
./2DIBLBM

cd ../../../IB-LBM
# bash bash/analysis.sh $WORKING_DIR
# sbatch analysis.sh $WORKING_DIR
