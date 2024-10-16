# IB-LBM

This repository aims at applying immersed boundary(IB) method to lattice Boltzmann method
(LBM) simulation for active fluid flow to investigate the effect of fluid confinement
geometry to fluid flow dynamics.

## The Goal

The goal is to develop a python-based interface for user-friendly LBM platform for active
fluid community, specifically, active nematics.

The idea is to export a .stl file from CAD modeling, through a converter script, input
to the IBLBM simulation script about the immersed boundary object information.

## How to start

This platform is also compatible with High Performance Computing servers.

To start, it is recommended to use VSCode as your text editor.

`ssh your@hostname` to server , under the home directory,

`git clone https://github.com/Daniel-Tianxing-Weng/IB-LBM.git`

After the repo is cloned, make directory to store the generated data,

`mkdir -p ~/Data`

Additionally, configure the virtual environment

`cd ~`
`module load python`
`python -m venv .venv`
`source .venv/bin/activate`
`pip install numpy matplotlib pandas jupyter`

Then in VSCode, manually enter path for python interpreter `/home/.venv/bin/python`
for Jupyter notebook to run on the kernel.
