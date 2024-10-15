#!/bin/bash
#SBATCH -J analysis
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --mem 10G
#SBATCH -p short
#SBATCH -t 06:00:00
#SBATCH --output=analysis_%j.out
#SBATCH --error=analysis_%j.err

scp analysis/velocity_correlation.py $1/velocity_correlation.py
scp analysis/nematic_correlation.py $1/nematic_correlation.py
scp analysis/organize_data.py $1/organize_data.py
# scp analysis/plotter.py $1/plotter.py

source ../.venv/bin/activate

cd $1

python3 -Wi organize_data.py
python3 -Wi velocity_correlation.py
python3 -Wi nematic_correlation.py


mkdir -p movies
module load ffmpeg
ffmpeg -r 24 -i frames/flow_t%05d.png -vcodec mpeg4 -y movies/flow.mp4
ffmpeg -r 24 -i frames/orientation_t%05d.png -vcodec mpeg4 -y movies/orientation.mp4
