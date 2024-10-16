#!/bin/bash
#SBATCH -J lbm
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --mem 10G
#SBATCH -p short
#SBATCH -t 06:00:00
#SBATCH --output=lbm_%j.out
#SBATCH --error=lbm_%j.err


WORKING_DIR="../Data/${1}/${2}"
mkdir -p $WORKING_DIR

mv temp/${2}/SimulationParameters.txt $WORKING_DIR/SimulationParameters.txt

g++ -std=c++17 -o $WORKING_DIR/3DLBM src/LBM_3D_cavity.cpp
cd $WORKING_DIR
./3DLBM 

cd ../../../IB-LBM
bash bash/analysis.sh $WORKING_DIR
# sbatch analysis.sh $WORKING_DIR
