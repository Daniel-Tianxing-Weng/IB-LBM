/* This code accompanies
 *   The Lattice Boltzmann Method: Principles and Practice
 *   T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, E.M. Viggen
 *   ISBN 978-3-319-44649-3 (Electronic) 
 *        978-3-319-44647-9 (Print)
 *   http://www.springer.com/978-3-319-44647-9
 *
 * This code is provided under the MIT license. See LICENSE.txt.
 *
 * Author: Timm Krüger
 *
 */

/* This is an example 3D lattice Boltzmann method code.
 * It uses the D3Q15 lattice with Guo's forcing term.
 * The purpose is to simulate cavity flow with(out) rigid walls simulated by IBM nodes.
 *
 * The D3Q15 lattice velocities are defined according to the following scheme:
 * index:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
 * ----------------------------------------------------
 * x:       0 +1 -1  0  0  0  0 +1 -1 +1 -1 +1 -1 -1 +1
 * y:       0  0  0 +1 -1  0  0 +1 -1 +1 -1 -1 +1 +1 -1
 * z:       0  0  0  0  0 +1 -1 +1 -1 -1 +1 +1 -1 +1 -1
 *
 *    5  3    ^z  y
 *    | /     | /  
 * 2- 0 -1    ---->x
 *  / |     
 * 4  6 
 * 
 *         -> vel_back
 *       --------
 *      /      / |      ^z  y
 *     --------         | /  
 *     |      | /       ---->x
 *     --------
 *  (0,0,0)
 * 
 */

// *********************
// PREPROCESSOR COMMANDS
// *********************

#include <vector>   // vector containers
#include <cmath>    // mathematical library
#include <iostream> // for the use of 'cout'
#include <fstream>  // file streams
#include <sstream>  // string streams
#include <cstdlib>  // standard library
#include <chrono>   // time library
#include <random>   // random generator library

#include "tqdm.h"

#define SQ(x) ((x) * (x)) // square function; replaces SQ(x) by ((x) * (x)) in the code


// *********************
// SIMULATION PARAMETERS
// *********************

// These are the relevant simulation parameters that can be changed by the user.
// If a bottom or top wall shall move in negative x-direction, a negative velocity has to be specified.
// Moving walls and gravity can be switched on simultaneously.

// Basic fluid/lattice properties
int Nx; // = 61; // number of lattice nodes along the x-axis (including two wall nodes)
int Ny; // = 61; // number of lattice nodes along the y-axis (including two wall nodes)
int Nz; // = 11; // number of lattice nodes along the z-axis (including two wall nodes)
int t_num; // = 6000; // number of time steps (running from 1 to t_num)
double vel_back; // = 5e-4; // velocity of the back wall (in positive x-direction)
double Gamma; // = 2; // relaxation time for Q (have to be much smaller than tau)
double L; // = 0.001; // Frank constant for one-constant approximation of elastic free energy
double U; // = 0.7; // controls isotropic-nematic transition, at U = 1 (above nematic, below isotropic)
double zeta; // = 0; // activity parameter
double A; // = 1e-4; // controls strength of nematic ordering

double a; // free energy parameter in Q^2 term
double b; // free energy parameter in Q^3 term
double c; // free energy parameter in Q^4 term


const double tau = 0.9; // relaxation time
const int t_disk = 1000; // disk write time step (data will be written to the disk every t_disk step)
const int t_info = 1000; // info time step (screen message will be printed every t_info step)
const double nu = (tau - 0.5) / 3; // viscosity
const double xi = 1; // flow aligning parameter

// ***************************************
// DECLARE ADDITIONAL VARIABLES AND ARRAYS
// ***************************************

const double omega = 1. / tau; // relaxation frequency (inverse of relaxation time)
double ****pop, ****pop_old; // LBM populations (old and new)
double ***density; // fluid density
double ***velocity_x; // fluid velocity (x-component)
double ***velocity_y; // fluid velocity (y-component)
double ***velocity_z; // fluid velocity (z-component)
double ***force_x; // fluid force (x-component)
double ***force_y; // fluid force (y-component)
double ***force_z; // fluid force (z-component)
double ****stress; // fluid stress ()
double ****Q; // order parameter (0-xx, 1-yy, 2-xy, 3-yz, 4-xz)
double force_latt[15]; // lattice force term entering the lattice Boltzmann equation
double pop_eq[15]; // equilibrium populations
const double weight[15] = {2./9., 1./9., 1./9., 1./9., 1./9., 1./9., 1./9., 1./72., 1./72., 1./72., 1./72., 1./72., 1./72., 1./72., 1./72.}; // lattice weights

// *****************
// DECLARE FUNCTIONS
// *****************

void initialize(); // allocate memory and initialize variables
void LBM(); // perform LBM operations
void Q_evolution(); // perform Euler integration
void force(); // compute force from Q configuration
void momenta(); // compute fluid density and velocity from the populations
void equilibrium(double, double, double, double); // compute the equilibrium populations from the fluid density and velocity
void write_fluid_profile(int); // write lattice velocity profile to disk

// *************
// MAIN FUNCTION
// *************

// This is the main function, containing the simulation initialization and the simulation loop.
// Overview of simulation algorithm:
// 1) compute the node forces based on the object's deformation
// 2) spread the node forces to the fluid lattice
// 3) update the fluid state via LBM
// 4) interpolate the fluid velocity to the object nodes
// 5) update node positions
// 6) if desired, write data to disk and report status

int main( int argc, char *argv[]) {

  if ( argc != 11 ) // argc should be 11 for correct execution
    // We print argv[0] assuming it is the program name
    std::cout<<"usage: "<< argv[0] <<" <Nx> <Ny> <Nz> <t_num> <vel_back> <Gamma> <L> <U> <A> <zeta>\n";
  else {
    Nx = atoi(argv[1]); // number of lattice nodes along the x-axis (including two wall nodes)
    Ny = atoi(argv[2]); // number of lattice nodes along the y-axis (including two wall nodes)
    Nz = atoi(argv[3]); // number of lattice nodes along the z-axis (including two wall nodes)
    t_num = atoi(argv[4]); // number of time steps (running from 1 to t_num)
    vel_back = atof(argv[5]); // velocity of the back wall (in positive x-direction)
    Gamma = atof(argv[6]); // relaxation frequency for Q (AGamma have to be much smaller than 1)
    L = atof(argv[7]); // Frank constant for one-constant approximation of elastic free energy
    U = atof(argv[8]); // controls isotropic-nematic transition, at U = 1 (above nematic, below isotropic)
    A = atof(argv[9]); // controls strength of nematic ordering
    zeta = -atof(argv[10]); // activity parameter
  }

  initialize(); // allocate memory and initialize variables

  // std::cout << "a: " << a << " , b: " << b << " , c: " << c << std::endl;

  const auto start = std::chrono::high_resolution_clock::now(); // start clock

  std::cout << "Starting simulation:" << std::endl;

  tqdm bar;

  for(int t = 1; t <= t_num; ++t) { // run over all times between 1 and t_num
    
    bar.progress(t,t_num);

    LBM(); // perform collision, propagation, and bounce-back

    Q_evolution(); // perform Euler integration on the Q variable
    
    // Write fluid and particle to VTK files
    if(t % t_disk == 0) {
      write_fluid_profile(t);
    }
  }

  bar.finish();

  // Report successful end of simulation
  const auto stop = std::chrono::high_resolution_clock::now();   // floating-point duration: no duration_cast needed
  auto duration = std::chrono::duration_cast<std::chrono::minutes>(stop - start);
  std::cout << "Simulation complete, takes " 
       << duration.count() << " minutes" << std::endl;

  return 0;
}

// ****************************************
// ALLOCATE MEMORY AND INITIALIZE VARIABLES
// ****************************************

// The memory for lattice variables (populations, density, velocity, forces) is allocated.
// The variables are initialized.

void initialize() {
  // Create folders, delete data file
  // Make sure that the VTK folders exist.
  // Old file data.dat is deleted, if existing.
  // string FILE_PATH = "LID_SPEED = " + to_string(vel_back) + "A = " + to_string(A) + "Gamma = " + to_string(Gamma) + "L = " + to_string(L) + "zeta = " + to_string(zeta);
  int ignore; // ignore return value of system calls
  ignore = system("mkdir -p data"); // create folder if not existing
  ignore = system("rm -f data/fluid_t*.dat"); // delete file if existing

  // write AnalysisParameters file
  std::stringstream output_filename;
  output_filename << "AnalysisParameters.dat";
  std::ofstream output_file;
  output_file.open(output_filename.str().c_str());
  
  // Write Parameters.
  output_file << "Nx Ny Nz VISCOSITY LID_SPEED GAMMA ZETA NEMATIC_DENSITY NEMATIC_STRENGTH FRANK_CONSTANT TOTAL_TIME DISK_TIME\n";
  output_file << Nx << " " << Ny << " " << Nz << " " << nu << " " << vel_back << " " << Gamma << " " << zeta << " " << A << " " << U << " " << L << " " << t_num << " " << t_disk << "\n";

  output_file.close();

  a = A * 3. * (1 - U) / 2; // free energy parameter in Q^2 term
  b = A * 9. * U / 2; // free energy parameter in Q^3 term
  c = A * 9. * U; // free energy parameter in Q^4 term
  
  // Allocate memory for the fluid density, velocity, and force, and director
  // Initialize the fluid density and velocity. Start with unit density and zero velocity.
  density = new double**[Nx];
  velocity_x = new double**[Nx];
  velocity_y = new double**[Nx];
  velocity_z = new double**[Nx];
  force_x = new double**[Nx];
  force_y = new double**[Nx];
  force_z = new double**[Nx];

  for(int X = 0; X < Nx; ++X) {
    density[X] = new double*[Ny];
    velocity_x[X] = new double*[Ny];
    velocity_y[X] = new double*[Ny];
    velocity_z[X] = new double*[Ny];
    force_x[X] = new double*[Ny];
    force_y[X] = new double*[Ny];
    force_z[X] = new double*[Ny];

    for(int Y = 0; Y < Ny; ++Y) {
      density[X][Y] = new double[Nz];
      velocity_x[X][Y] = new double[Nz];
      velocity_y[X][Y] = new double[Nz];
      velocity_z[X][Y] = new double[Nz];
      force_x[X][Y] = new double[Nz];
      force_y[X][Y] = new double[Nz];
      force_z[X][Y] = new double[Nz];
      for(int Z = 0; Z < Nz; ++Z) {
        density[X][Y][Z] = 1;
        // back wall y = Ny - 1, velocity_x = vel_back
        if(Y == (Ny - 1)){ // && Z != 0 && Z != (Nz -1) && X != 0 && X != (Nx - 1)) {
          velocity_x[X][Y][Z] = vel_back;
        }
        else{
          velocity_x[X][Y][Z] = 0;
        }
        velocity_y[X][Y][Z] = 0;
        velocity_z[X][Y][Z] = 0;

        force_x[X][Y][Z] = 0;
        force_y[X][Y][Z] = 0;
        force_z[X][Y][Z] = 0;
      }
    }
  }

  stress = new double***[9];
  for(int i = 0; i < 9; ++i) {
    stress[i] = new double**[Nx];
    for(int X = 0; X < Nx; ++X) {
      stress[i][X] = new double*[Ny];
      for(int Y = 0; Y < Ny; ++Y) {
        stress[i][X][Y] = new double[Nz];
        for(int Z = 0; Z < Nz; ++Z) {
          stress[i][X][Y][Z] = 0;
        }
      }
    }
  }

  Q = new double***[5];
  for(int c_i = 0; c_i < 5; ++c_i) {
    Q[c_i] = new double**[Nx];
    for(int X = 0; X < Nx; ++X) {
      Q[c_i][X] = new double*[Ny];
      for(int Y = 0; Y < Ny; ++Y) {
        Q[c_i][X][Y] = new double[Nz];
      }
    }
  }

  // Random initial conditions for director
  for(int X = 0; X < Nx; ++X) {
    for(int Y = 0; Y < Ny; ++Y) {
      for(int Z = 0; Z < Nz; ++Z) {

        const double theta = M_PI * (double)rand() / (double)RAND_MAX;
        const double phi = 2 * M_PI * (double)rand() / (double)RAND_MAX;

        const double initial_scalar_order = 0.01 * (double)rand() / (double)RAND_MAX;
        const double initial_nx = sin(theta) * cos(phi);
        const double initial_ny = sin(theta) * sin(phi);
        const double initial_nz = cos(theta);

        Q[0][X][Y][Z] = initial_scalar_order * (SQ(initial_nx) - 1./3);
        Q[1][X][Y][Z] = initial_scalar_order * (SQ(initial_ny) - 1./3);
        Q[2][X][Y][Z] = initial_scalar_order * (initial_nx * initial_ny);
        Q[3][X][Y][Z] = initial_scalar_order * (initial_ny * initial_nz);
        Q[4][X][Y][Z] = initial_scalar_order * (initial_nz * initial_nx);
      }
    }
  }        

  // Update force from director field
  force();

  // Allocate memory for the populations
  pop = new double***[15];
  pop_old = new double***[15];

  for(int c_i = 0; c_i < 15; ++c_i) {
    pop[c_i] = new double**[Nx];
    pop_old[c_i] = new double**[Nx];

    for(int X = 0; X < Nx; ++X) {
      pop[c_i][X] = new double*[Ny];
      pop_old[c_i][X] = new double*[Ny];

      for(int Y = 0; Y < Ny; ++Y) {
        pop[c_i][X][Y] = new double[Nz];
        pop_old[c_i][X][Y] = new double[Nz];

        for(int Z = 0; Z < Nz; ++Z) {
          pop[c_i][X][Y][Z] = 0;
          pop_old[c_i][X][Y][Z] = 0;
        }
      }
    }
  }

  // Initialize the populations. Use the equilibrium populations corresponding to the initialized fluid density and velocity.
  // Can also improve this by incorporating the initial forcing scheme.
  for(int X = 0; X < Nx; ++X) {
    for(int Y = 0; Y < Ny; ++Y) {
      for(int Z = 0; Z < Nz; ++Z) {
        equilibrium(density[X][Y][Z], velocity_x[X][Y][Z] - 0.5 * (force_x[X][Y][Z]) / density[X][Y][Z], velocity_y[X][Y][Z] - 0.5 * (force_y[X][Y][Z]) / density[X][Y][Z], velocity_z[X][Y][Z] - 0.5 * (force_z[X][Y][Z]) / density[X][Y][Z]);
        for(int c_i = 0; c_i < 15; ++c_i) {
          pop_old[c_i][X][Y][Z] = pop_eq[c_i];
          pop[c_i][X][Y][Z] = pop_eq[c_i];
        }
      }
    }
  }

  return;
}

// ***************
// COMPUTE FORCES
// ***************

// This function computes forces based on passive and active stresses

void force() {

  // initialize the stress tensor to fill in
  // stress = σ + τ 
  // τ = Q H - H Q
  // σ = - ξ (Q + 1/3) H - ξ H (Q + 1/3) + 2 ξ (Q + 1/3) tr(QH) - L nabla Q odot nabla Q
  // 0-xx, 1-xy, 2-xz, 3-yx, 4-yy, 5-yz, 6-zx, 7-zy, 8-zz
  

  // Calculate stress based on stress = σ + τ 
  // τ = Q H - H Q
  // σ = - ξ (Q + 1/3) H - ξ H (Q + 1/3) + 2 ξ (Q + 1/3) tr(QH) - L nabla Q odot nabla Q
  for(int X = 1; X < Nx - 1; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      for(int Z = 1; Z < Nz - 1; ++Z) {

        // Calculate relaxation first from H = L ∆ Q - a Q + b (Q^2 - I/3 tr(Q^2)) - c Q tr(Q^2)
        double relaxation[5];
        const double trace_relaxation = (SQ(Q[0][X][Y][Z]) + SQ(Q[1][X][Y][Z]) + SQ(Q[2][X][Y][Z]) + SQ(Q[3][X][Y][Z]) + SQ(Q[4][X][Y][Z]) + Q[0][X][Y][Z] * Q[1][X][Y][Z]);
        relaxation[0] = L * (Q[0][X+1][Y][Z] + Q[0][X-1][Y][Z] + Q[0][X][Y+1][Z] + Q[0][X][Y-1][Z] + Q[0][X][Y][Z+1] + Q[0][X][Y][Z-1] - 6 * Q[0][X][Y][Z]) - (a + c * trace_relaxation) * Q[0][X][Y][Z] + b * (SQ(Q[0][X][Y][Z]) + SQ(Q[2][X][Y][Z]) + SQ(Q[4][X][Y][Z]) - 1. /3 * trace_relaxation);
        relaxation[1] = L * (Q[1][X+1][Y][Z] + Q[1][X-1][Y][Z] + Q[1][X][Y+1][Z] + Q[1][X][Y-1][Z] + Q[1][X][Y][Z+1] + Q[1][X][Y][Z-1] - 6 * Q[1][X][Y][Z]) - (a + c * trace_relaxation) * Q[1][X][Y][Z] + b * (SQ(Q[0][X][Y][Z]) + SQ(Q[2][X][Y][Z]) + SQ(Q[4][X][Y][Z]) - 1. /3 * trace_relaxation);
        relaxation[2] = L * (Q[2][X+1][Y][Z] + Q[2][X-1][Y][Z] + Q[2][X][Y+1][Z] + Q[2][X][Y-1][Z] + Q[2][X][Y][Z+1] + Q[2][X][Y][Z-1] - 6 * Q[2][X][Y][Z]) - (a + c * trace_relaxation) * Q[2][X][Y][Z] + b * (Q[0][X][Y][Z] * Q[2][X][Y][Z] + Q[1][X][Y][Z] * Q[2][X][Y][Z] + Q[3][X][Y][Z] * Q[4][X][Y][Z]);
        relaxation[3] = L * (Q[3][X+1][Y][Z] + Q[3][X-1][Y][Z] + Q[3][X][Y+1][Z] + Q[3][X][Y-1][Z] + Q[3][X][Y][Z+1] + Q[3][X][Y][Z-1] - 6 * Q[3][X][Y][Z]) - (a + c * trace_relaxation) * Q[3][X][Y][Z] + b * (Q[2][X][Y][Z] * Q[4][X][Y][Z] - Q[0][X][Y][Z] * Q[3][X][Y][Z]);
        relaxation[4] = L * (Q[4][X+1][Y][Z] + Q[4][X-1][Y][Z] + Q[4][X][Y+1][Z] + Q[4][X][Y-1][Z] + Q[4][X][Y][Z+1] + Q[4][X][Y][Z-1] - 6 * Q[4][X][Y][Z]) - (a + c * trace_relaxation) * Q[4][X][Y][Z] + b * (Q[2][X][Y][Z] * Q[3][X][Y][Z] - Q[1][X][Y][Z] * Q[4][X][Y][Z]);

        // anti-symmetric 0-yz, 1-zx, 2-xy
        // τ = Q H - H Q
        double stress_antisymmetric[3];
        stress_antisymmetric[0] = Q[2][X][Y][Z] * relaxation[4] - relaxation[2] * Q[4][X][Y][Z] + relaxation[3] * (2 * Q[1][X][Y][Z] + Q[0][X][Y][Z]) - Q[3][X][Y][Z] * (2 * relaxation[1] + relaxation[0]);
        stress_antisymmetric[1] = Q[3][X][Y][Z] * relaxation[2] - relaxation[3] * Q[2][X][Y][Z] - relaxation[4] * (2 * Q[0][X][Y][Z] + Q[1][X][Y][Z]) + Q[4][X][Y][Z] * (2 * relaxation[0] + relaxation[1]);
        stress_antisymmetric[2] = Q[4][X][Y][Z] * relaxation[3] - relaxation[4] * Q[3][X][Y][Z] + relaxation[2] * (Q[0][X][Y][Z] - Q[1][X][Y][Z]) - Q[2][X][Y][Z] * (relaxation[0] - relaxation[1]);

        // 0-xx, 1-yy, 2-xy, 3-yz, 4-zx, 5-zz
        // σ = ζ Q - ξ (Q + 1/3) H - ξ H (Q + 1/3) + 2 ξ (Q + 1/3) tr(QH) 
        //   = ζ Q - 2/3 ξ H + 2 ξ (Q + 1/3) tr(QH) - ξ (Q H + H Q) 
        double stress_symmetric[5];
        const double trace_stress = 2 * (Q[0][X][Y][Z] * relaxation[0] + Q[1][X][Y][Z] * relaxation[1] + Q[2][X][Y][Z] * relaxation[2] + Q[3][X][Y][Z] * relaxation[3] + Q[4][X][Y][Z] * relaxation[4]) + Q[0][X][Y][Z] * relaxation[1] + Q[1][X][Y][Z] * relaxation[0];
        stress_symmetric[0] = zeta * Q[0][X][Y][Z] - xi * 2./3 * relaxation[0] + 2 * xi * (Q[0][X][Y][Z] + 1./3) * trace_stress - xi * 2 * (Q[0][X][Y][Z] * relaxation[0] + Q[2][X][Y][Z] * relaxation[2] + Q[4][X][Y][Z] * relaxation[4]);
        stress_symmetric[1] = zeta * Q[1][X][Y][Z] - xi * 2./3 * relaxation[1] + 2 * xi * (Q[1][X][Y][Z] + 1./3) * trace_stress - xi * 2 * (Q[1][X][Y][Z] * relaxation[1] + Q[2][X][Y][Z] * relaxation[2] + Q[3][X][Y][Z] * relaxation[3]);
        stress_symmetric[2] = zeta * Q[2][X][Y][Z] - xi * 2./3 * relaxation[2] + 2 * xi * (Q[2][X][Y][Z]       ) * trace_stress - xi * ((Q[0][X][Y][Z] + Q[1][X][Y][Z]) * relaxation[2] + Q[2][X][Y][Z] * (relaxation[0] + relaxation[1]) + Q[4][X][Y][Z] * relaxation[3] + Q[3][X][Y][Z] * relaxation[4]);
        stress_symmetric[3] = zeta * Q[3][X][Y][Z] - xi * 2./3 * relaxation[3] + 2 * xi * (Q[3][X][Y][Z]       ) * trace_stress - xi * (Q[2][X][Y][Z] * relaxation[4] + Q[4][X][Y][Z] * relaxation[2] - Q[0][X][Y][Z] * relaxation[3] - Q[3][X][Y][Z] * relaxation[0]);
        stress_symmetric[4] = zeta * Q[4][X][Y][Z] - xi * 2./3 * relaxation[4] + 2 * xi * (Q[4][X][Y][Z]       ) * trace_stress - xi * (Q[2][X][Y][Z] * relaxation[3] + Q[3][X][Y][Z] * relaxation[2] - Q[1][X][Y][Z] * relaxation[4] - Q[4][X][Y][Z] * relaxation[1]);
        
        // 0-xx, 1-yy, 2-xy, 3-yz, 4-zx, 5-zz
        // σ_ij = - L ∂_j Q_km  ∂_i Q_km
        double stress_elastic[6];
        stress_elastic[0] = -L * 2 * (SQ((Q[0][X+1][Y][Z] - Q[0][X-1][Y][Z])/2) + SQ((Q[1][X+1][Y][Z] - Q[1][X-1][Y][Z])/2) + SQ((Q[2][X+1][Y][Z] - Q[2][X-1][Y][Z])/2) + SQ((Q[3][X+1][Y][Z] - Q[3][X-1][Y][Z])/2) + SQ((Q[4][X+1][Y][Z] - Q[4][X-1][Y][Z])/2) + ((Q[0][X+1][Y][Z] - Q[0][X-1][Y][Z])/2) * (Q[1][X+1][Y][Z] - Q[1][X-1][Y][Z])/2);
        stress_elastic[1] = -L * 2 * (SQ((Q[0][X][Y+1][Z] - Q[0][X][Y-1][Z])/2) + SQ((Q[1][X][Y+1][Z] - Q[1][X][Y-1][Z])/2) + SQ((Q[2][X][Y+1][Z] - Q[2][X][Y-1][Z])/2) + SQ((Q[3][X][Y+1][Z] - Q[3][X][Y-1][Z])/2) + SQ((Q[4][X][Y+1][Z] - Q[4][X][Y-1][Z])/2) + ((Q[0][X][Y+1][Z] - Q[0][X][Y-1][Z])/2) * (Q[1][X][Y+1][Z] - Q[1][X][Y-1][Z])/2);
        stress_elastic[2] = -L * (2 * ((Q[0][X+1][Y][Z] - Q[0][X-1][Y][Z])/2 * (Q[0][X][Y+1][Z] - Q[0][X][Y-1][Z])/2 + (Q[1][X+1][Y][Z] - Q[1][X-1][Y][Z])/2 * (Q[1][X][Y+1][Z] - Q[1][X][Y-1][Z])/2 + (Q[2][X+1][Y][Z] - Q[2][X-1][Y][Z])/2 * (Q[2][X][Y+1][Z] - Q[2][X][Y-1][Z])/2 + (Q[3][X+1][Y][Z] - Q[3][X-1][Y][Z])/2 * (Q[3][X][Y+1][Z] - Q[3][X][Y-1][Z])/2 + (Q[4][X+1][Y][Z] - Q[4][X-1][Y][Z])/2 * (Q[4][X][Y+1][Z] - Q[4][X][Y-1][Z])/2) + (Q[0][X+1][Y][Z] - Q[0][X-1][Y][Z])/2 * (Q[1][X][Y+1][Z] - Q[1][X][Y-1][Z])/2 + (Q[1][X+1][Y][Z] - Q[1][X-1][Y][Z])/2 * (Q[0][X][Y+1][Z] - Q[0][X][Y-1][Z])/2);
        stress_elastic[3] = -L * (2 * ((Q[0][X][Y+1][Z] - Q[0][X][Y-1][Z])/2 * (Q[0][X][Y][Z+1] - Q[0][X][Y][Z-1])/2 + (Q[1][X][Y+1][Z] - Q[1][X][Y-1][Z])/2 * (Q[1][X][Y][Z+1] - Q[1][X][Y][Z-1])/2 + (Q[2][X][Y+1][Z] - Q[2][X][Y-1][Z])/2 * (Q[2][X][Y][Z+1] - Q[2][X][Y][Z-1])/2 + (Q[3][X][Y+1][Z] - Q[3][X][Y-1][Z])/2 * (Q[3][X][Y][Z+1] - Q[3][X][Y][Z-1])/2 + (Q[4][X][Y+1][Z] - Q[4][X][Y-1][Z])/2 * (Q[4][X][Y][Z+1] - Q[4][X][Y][Z-1])/2) + (Q[0][X][Y+1][Z] - Q[0][X][Y-1][Z])/2 * (Q[1][X][Y][Z+1] - Q[1][X][Y][Z-1])/2 + (Q[1][X][Y+1][Z] - Q[1][X][Y-1][Z])/2 * (Q[0][X][Y][Z+1] - Q[0][X][Y][Z-1])/2);
        stress_elastic[4] = -L * (2 * ((Q[0][X][Y][Z+1] - Q[0][X][Y][Z-1])/2 * (Q[0][X+1][Y][Z] - Q[0][X-1][Y][Z])/2 + (Q[1][X][Y][Z+1] - Q[1][X][Y][Z-1])/2 * (Q[1][X+1][Y][Z] - Q[1][X-1][Y][Z])/2 + (Q[2][X][Y][Z+1] - Q[2][X][Y][Z-1])/2 * (Q[2][X+1][Y][Z] - Q[2][X-1][Y][Z])/2 + (Q[3][X][Y][Z+1] - Q[3][X][Y][Z-1])/2 * (Q[3][X+1][Y][Z] - Q[3][X-1][Y][Z])/2 + (Q[4][X][Y][Z+1] - Q[4][X][Y][Z-1])/2 * (Q[4][X+1][Y][Z] - Q[4][X-1][Y][Z])/2) + (Q[0][X][Y][Z+1] - Q[0][X][Y][Z-1])/2 * (Q[1][X+1][Y][Z] - Q[1][X-1][Y][Z])/2 + (Q[1][X][Y][Z+1] - Q[1][X][Y][Z-1])/2 * (Q[0][X+1][Y][Z] - Q[0][X-1][Y][Z])/2);
        stress_elastic[5] = -L * 2 * (SQ((Q[0][X][Y][Z+1] - Q[0][X][Y][Z-1])/2) + SQ((Q[1][X][Y][Z+1] - Q[1][X][Y][Z-1])/2) + SQ((Q[2][X][Y][Z+1] - Q[2][X][Y][Z-1])/2) + SQ((Q[3][X][Y][Z+1] - Q[3][X][Y][Z-1])/2) + SQ((Q[4][X][Y][Z+1] - Q[4][X][Y][Z-1])/2) + ((Q[0][X][Y][Z+1] - Q[0][X][Y][Z-1])/2) * (Q[1][X][Y][Z+1] - Q[1][X][Y][Z-1])/2);


        // Calculate stress
        // 0-xx, 1-xy, 2-xz, 3-yx, 4-yy, 5-yz, 6-zx, 7-zy, 8-zz
        stress[0][X][Y][Z] = stress_symmetric[0] + stress_elastic[0];
        stress[1][X][Y][Z] = stress_symmetric[2] + stress_elastic[2] + stress_antisymmetric[2];
        stress[2][X][Y][Z] = stress_symmetric[4] + stress_elastic[4] - stress_antisymmetric[1];
        stress[3][X][Y][Z] = stress_symmetric[2] + stress_elastic[2] - stress_antisymmetric[2];
        stress[4][X][Y][Z] = stress_symmetric[1] + stress_elastic[1];
        stress[5][X][Y][Z] = stress_symmetric[3] + stress_elastic[3] - stress_antisymmetric[0];
        stress[6][X][Y][Z] = stress_symmetric[4] + stress_elastic[4] + stress_antisymmetric[1];
        stress[7][X][Y][Z] = stress_symmetric[3] + stress_elastic[3] + stress_antisymmetric[0];
        stress[8][X][Y][Z] = - stress_symmetric[0] - stress_symmetric[1] + stress_elastic[5];
      }
    }
  }

  // update force
  for(int X = 1; X < Nx - 1; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      for(int Z = 1; Z < Nz - 1; ++Z) {
        force_x[X][Y][Z] = -(stress[0][X+1][Y][Z] - stress[0][X-1][Y][Z] + stress[1][X][Y+1][Z] - stress[1][X][Y-1][Z] + stress[2][X][Y][Z+1] - stress[2][X][Y][Z-1]) / 2;
        force_y[X][Y][Z] = -(stress[3][X+1][Y][Z] - stress[3][X-1][Y][Z] + stress[4][X][Y+1][Z] - stress[4][X][Y-1][Z] + stress[5][X][Y][Z+1] - stress[5][X][Y][Z-1]) / 2;
        force_z[X][Y][Z] = -(stress[6][X+1][Y][Z] - stress[6][X-1][Y][Z] + stress[7][X][Y+1][Z] - stress[7][X][Y-1][Z] + stress[8][X][Y][Z+1] - stress[8][X][Y][Z-1]) / 2;
      }
    }
  }


  return;
}

// ***************
// UPDATE Q FIELD
// ***************

// This function computes the order director field based on Euler integration
// 

void Q_evolution() {
  // double ****Q_new;
  // Q_new = new double***[5];
  // for(int i = 0; i < 5; ++i) {
  //   Q_new[i] = new double**[Nx];
  //   for(int X = 0; X < Nx; ++X) {
  //     Q_new[i][X] = new double*[Ny];
  //     for(int Y = 0; Y < Ny; ++Y) {
  //       Q_new[i][X][Y] = new double[Nz];
  //       for(int Z = 0; Z < Nz; ++Z) {
  //         Q_new[i][X][Y][Z] = 0;
  //       }
  //     }
  //   }
  // }

  for(int X = 1; X < Nx - 1; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      for(int Z = 1; Z < Nz - 1; ++Z) {

        // Calculate relaxation first from H = L ∆ Q - a Q + b (Q^2 - I/3 tr(Q^2)) - c Q tr(Q^2)
        double relaxation[5];
        const double trace = (SQ(Q[0][X][Y][Z]) + SQ(Q[1][X][Y][Z]) + SQ(Q[2][X][Y][Z]) + SQ(Q[3][X][Y][Z]) + SQ(Q[4][X][Y][Z]) + Q[0][X][Y][Z] * Q[1][X][Y][Z]);
        relaxation[0] = L * (Q[0][X+1][Y][Z] + Q[0][X-1][Y][Z] + Q[0][X][Y+1][Z] + Q[0][X][Y-1][Z] + Q[0][X][Y][Z+1] + Q[0][X][Y][Z-1] - 6 * Q[0][X][Y][Z]) - (a + c * trace) * Q[0][X][Y][Z] + b * (SQ(Q[0][X][Y][Z]) + SQ(Q[2][X][Y][Z]) + SQ(Q[4][X][Y][Z]) - 1. /3 * trace);
        relaxation[1] = L * (Q[1][X+1][Y][Z] + Q[1][X-1][Y][Z] + Q[1][X][Y+1][Z] + Q[1][X][Y-1][Z] + Q[1][X][Y][Z+1] + Q[1][X][Y][Z-1] - 6 * Q[1][X][Y][Z]) - (a + c * trace) * Q[1][X][Y][Z] + b * (SQ(Q[0][X][Y][Z]) + SQ(Q[2][X][Y][Z]) + SQ(Q[4][X][Y][Z]) - 1. /3 * trace);
        relaxation[2] = L * (Q[2][X+1][Y][Z] + Q[2][X-1][Y][Z] + Q[2][X][Y+1][Z] + Q[2][X][Y-1][Z] + Q[2][X][Y][Z+1] + Q[2][X][Y][Z-1] - 6 * Q[2][X][Y][Z]) - (a + c * trace) * Q[2][X][Y][Z] + b * (Q[0][X][Y][Z] * Q[2][X][Y][Z] + Q[1][X][Y][Z] * Q[2][X][Y][Z] + Q[3][X][Y][Z] * Q[4][X][Y][Z]);
        relaxation[3] = L * (Q[3][X+1][Y][Z] + Q[3][X-1][Y][Z] + Q[3][X][Y+1][Z] + Q[3][X][Y-1][Z] + Q[3][X][Y][Z+1] + Q[3][X][Y][Z-1] - 6 * Q[3][X][Y][Z]) - (a + c * trace) * Q[3][X][Y][Z] + b * (Q[2][X][Y][Z] * Q[4][X][Y][Z] - Q[0][X][Y][Z] * Q[3][X][Y][Z]);
        relaxation[4] = L * (Q[4][X+1][Y][Z] + Q[4][X-1][Y][Z] + Q[4][X][Y+1][Z] + Q[4][X][Y-1][Z] + Q[4][X][Y][Z+1] + Q[4][X][Y][Z-1] - 6 * Q[4][X][Y][Z]) - (a + c * trace) * Q[4][X][Y][Z] + b * (Q[2][X][Y][Z] * Q[3][X][Y][Z] - Q[1][X][Y][Z] * Q[4][X][Y][Z]);

        // Calculate S(nabla u, Q) = ( ξ D + ω ) ( Q + 1/3 ) + ( Q + 1/3 )( ξ D - ω ) - 2 ξ ( Q + 1/3 ) tr(Q nabla u)
        double advection[5];
        double strain_rate[5];
        double vorticity[3];

        // D with central difference
        strain_rate[0] = (velocity_x[X+1][Y  ][Z  ] - velocity_x[X-1][Y  ][Z  ]) / 2;
        strain_rate[1] = (velocity_y[X  ][Y+1][Z  ] - velocity_y[X  ][Y-1][Z  ]) / 2;
        strain_rate[2] = (velocity_x[X  ][Y+1][Z  ] - velocity_x[X  ][Y-1][Z  ] + velocity_y[X+1][Y  ][Z  ] - velocity_y[X-1][Y  ][Z  ]) / 4;
        strain_rate[3] = (velocity_y[X  ][Y  ][Z+1] - velocity_y[X  ][Y  ][Z-1] + velocity_z[X  ][Y+1][Z  ] - velocity_z[X  ][Y-1][Z  ]) / 4;
        strain_rate[4] = (velocity_x[X  ][Y  ][Z+1] - velocity_x[X  ][Y  ][Z-1] + velocity_z[X+1][Y  ][Z  ] - velocity_z[X-1][Y  ][Z  ]) / 4;

        vorticity[0] = (velocity_z[X][Y+1][Z] - velocity_z[X][Y-1][Z]) / 2 - (velocity_y[X][Y][Z+1] - velocity_y[X][Y][Z-1]) / 2;
        vorticity[1] = (velocity_x[X][Y][Z+1] - velocity_x[X][Y][Z-1]) / 2 - (velocity_z[X+1][Y][Z] - velocity_z[X-1][Y][Z]) / 2;
        vorticity[2] = (velocity_y[X+1][Y][Z] - velocity_y[X-1][Y][Z]) / 2 - (velocity_x[X][Y+1][Z] - velocity_x[X][Y-1][Z]) / 2;

        const double traceless_constraint = Q[0][X][Y][Z] * (2 * strain_rate[0] + strain_rate[1]) + Q[1][X][Y][Z] * (strain_rate[0] + 2 * strain_rate[1]) + Q[2][X][Y][Z] * 2 * strain_rate[2] + Q[3][X][Y][Z] * 2 * strain_rate[3] + Q[4][X][Y][Z] * 2 * strain_rate[4];

        advection[0] = (-2 * xi * (Q[0][X][Y][Z] + 1./3) * traceless_constraint) + 2./3 * xi * strain_rate[0] + xi * 2 * (strain_rate[0] * Q[0][X][Y][Z] + strain_rate[2] * Q[2][X][Y][Z] + strain_rate[4] * Q[4][X][Y][Z]) + ( vorticity[2] * Q[2][X][Y][Z] - vorticity[1] * Q[4][X][Y][Z]);
        advection[1] = (-2 * xi * (Q[1][X][Y][Z] + 1./3) * traceless_constraint) + 2./3 * xi * strain_rate[1] + xi * 2 * (strain_rate[1] * Q[1][X][Y][Z] + strain_rate[2] * Q[2][X][Y][Z] + strain_rate[3] * Q[3][X][Y][Z]) + ( vorticity[0] * Q[3][X][Y][Z] - vorticity[2] * Q[2][X][Y][Z]);
        advection[2] = (-2 * xi * (Q[2][X][Y][Z]       ) * traceless_constraint) + 2./3 * xi * strain_rate[2] + xi * ((strain_rate[0] + strain_rate[1]) * Q[2][X][Y][Z] + strain_rate[2] * (Q[0][X][Y][Z] + Q[1][X][Y][Z]) + strain_rate[4] * Q[3][X][Y][Z] + strain_rate[3] * Q[4][X][Y][Z]) + ( vorticity[0] * Q[4][X][Y][Z] - vorticity[1] * Q[3][X][Y][Z] + vorticity[2] * (-Q[0][X][Y][Z] + Q[1][X][Y][Z])) / 2;
        advection[3] = (-2 * xi * (Q[3][X][Y][Z]       ) * traceless_constraint) + 2./3 * xi * strain_rate[3] + xi * (strain_rate[2] * Q[4][X][Y][Z] + strain_rate[4] * Q[2][X][Y][Z] - strain_rate[3] * Q[0][X][Y][Z] - strain_rate[0] * Q[3][X][Y][Z]) + ( vorticity[1] * Q[2][X][Y][Z] - vorticity[2] * Q[4][X][Y][Z] - vorticity[0] * ( Q[0][X][Y][Z] + 2 * Q[1][X][Y][Z])) / 2;
        advection[4] = (-2 * xi * (Q[4][X][Y][Z]       ) * traceless_constraint) + 2./3 * xi * strain_rate[4] + xi * (strain_rate[2] * Q[3][X][Y][Z] + strain_rate[3] * Q[2][X][Y][Z] - strain_rate[4] * Q[1][X][Y][Z] - strain_rate[1] * Q[4][X][Y][Z]) + ( vorticity[2] * Q[3][X][Y][Z] - vorticity[0] * Q[2][X][Y][Z] + vorticity[1] * ( 2 * Q[0][X][Y][Z] + Q[1][X][Y][Z])) / 2;


        // Calculate u dot nabla Q 
        double convection[5];
        convection[0] = velocity_x[X][Y][Z] * (Q[0][X+1][Y][Z] - Q[0][X-1][Y][Z]) / 2 + velocity_y[X][Y][Z] * (Q[0][X][Y+1][Z] - Q[0][X][Y-1][Z]) / 2 + velocity_z[X][Y][Z] * (Q[0][X][Y][Z+1] - Q[0][X][Y][Z-1]) / 2;
        convection[1] = velocity_x[X][Y][Z] * (Q[1][X+1][Y][Z] - Q[1][X-1][Y][Z]) / 2 + velocity_y[X][Y][Z] * (Q[1][X][Y+1][Z] - Q[1][X][Y-1][Z]) / 2 + velocity_z[X][Y][Z] * (Q[1][X][Y][Z+1] - Q[1][X][Y][Z-1]) / 2;
        convection[2] = velocity_x[X][Y][Z] * (Q[2][X+1][Y][Z] - Q[2][X-1][Y][Z]) / 2 + velocity_y[X][Y][Z] * (Q[2][X][Y+1][Z] - Q[2][X][Y-1][Z]) / 2 + velocity_z[X][Y][Z] * (Q[2][X][Y][Z+1] - Q[2][X][Y][Z-1]) / 2;
        convection[3] = velocity_x[X][Y][Z] * (Q[3][X+1][Y][Z] - Q[3][X-1][Y][Z]) / 2 + velocity_y[X][Y][Z] * (Q[3][X][Y+1][Z] - Q[3][X][Y-1][Z]) / 2 + velocity_z[X][Y][Z] * (Q[3][X][Y][Z+1] - Q[3][X][Y][Z-1]) / 2;
        convection[4] = velocity_x[X][Y][Z] * (Q[4][X+1][Y][Z] - Q[4][X-1][Y][Z]) / 2 + velocity_y[X][Y][Z] * (Q[4][X][Y+1][Z] - Q[4][X][Y-1][Z]) / 2 + velocity_z[X][Y][Z] * (Q[4][X][Y][Z+1] - Q[4][X][Y][Z-1]) / 2;


        // Perform Euler Integration based on Q equation:
        // Q (t + 1) = Q (t) + ∆t (gamma * H + S(nabla u, Q) - u dot nabla Q )
        Q[0][X][Y][Z] = Q[0][X][Y][Z] + Gamma * relaxation[0] + advection[0] - convection[0]; 
        Q[1][X][Y][Z] = Q[1][X][Y][Z] + Gamma * relaxation[1] + advection[1] - convection[1]; 
        Q[2][X][Y][Z] = Q[2][X][Y][Z] + Gamma * relaxation[2] + advection[2] - convection[2]; 
        Q[3][X][Y][Z] = Q[3][X][Y][Z] + Gamma * relaxation[3] + advection[3] - convection[3]; 
        Q[4][X][Y][Z] = Q[4][X][Y][Z] + Gamma * relaxation[4] + advection[4] - convection[4]; 
      }
    }
  }

  // Impose no-flux boundary conditions on the directors
  for(int X = 0; X < Nx; ++X) {
    for(int Y = 0; Y < Ny; ++Y) {
      for(int Z = 0; Z < Nz; ++Z) {
        // x = 0 & Nx - 1, 
        // nx = 0,
        // ny = 1
        if (X == 0) {
          Q[0][X][Y][Z] = Q[0][1][Y][Z];
          Q[1][X][Y][Z] = Q[1][1][Y][Z];
          Q[2][X][Y][Z] = Q[2][1][Y][Z];
          Q[3][X][Y][Z] = Q[3][1][Y][Z];
          Q[4][X][Y][Z] = Q[4][1][Y][Z];
        }
        if (X == Nx - 1) {
          Q[0][X][Y][Z] = Q[0][Nx-2][Y][Z];
          Q[1][X][Y][Z] = Q[1][Nx-2][Y][Z];
          Q[2][X][Y][Z] = Q[2][Nx-2][Y][Z];
          Q[3][X][Y][Z] = Q[3][Nx-2][Y][Z];
          Q[4][X][Y][Z] = Q[4][Nx-2][Y][Z];
        }
        // y = 0 & Ny - 1, 
        // ny = 0,
        // ny = 1
        if (Y == 0) {
          Q[0][X][Y][Z] = Q[0][X][1][Z];
          Q[1][X][Y][Z] = Q[1][X][1][Z];
          Q[2][X][Y][Z] = Q[2][X][1][Z];
          Q[3][X][Y][Z] = Q[3][X][1][Z];
          Q[4][X][Y][Z] = Q[4][X][1][Z];
        }
        if (Y == Ny - 1) {
          Q[0][X][Y][Z] = Q[0][X][Ny-2][Z];
          Q[1][X][Y][Z] = Q[1][X][Ny-2][Z];
          Q[2][X][Y][Z] = Q[2][X][Ny-2][Z];
          Q[3][X][Y][Z] = Q[3][X][Ny-2][Z];
          Q[4][X][Y][Z] = Q[4][X][Ny-2][Z];
        }
        // z = 0 & Nz - 1, 
        // nz = 0,
        // ny = 1
        if (Z == 0) {
          Q[0][X][Y][Z] = Q[0][X][Y][1];
          Q[1][X][Y][Z] = Q[1][X][Y][1];
          Q[2][X][Y][Z] = Q[2][X][Y][1];
          Q[3][X][Y][Z] = Q[3][X][Y][1];
          Q[4][X][Y][Z] = Q[4][X][Y][1];
        }
        if (Z == Nz - 1) {
          Q[0][X][Y][Z] = Q[0][X][Y][Nz-2];
          Q[1][X][Y][Z] = Q[1][X][Y][Nz-2];
          Q[2][X][Y][Z] = Q[2][X][Y][Nz-2];
          Q[3][X][Y][Z] = Q[3][X][Y][Nz-2];
          Q[4][X][Y][Z] = Q[4][X][Y][Nz-2];
        }
      }
    }
  }


  // // update Q and free up memory
  // Q = Q_new;
  // delete[] Q_new;
    return;
}

// *******************
// COMPUTE EQUILIBRIUM
// *******************

// This function computes the equilibrium populations from the fluid density and velocity.
// It computes the equilibrium only at a specific lattice node: Function has to be called at each lattice node.
// The standard quadratic equilibrium is used.

void equilibrium(double den, double vel_x, double vel_y, double vel_z) {
  pop_eq[0]  = weight[0]  * den * (1                                                                     - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[1]  = weight[1]  * den * (1 + 3 * (  vel_x                ) + 4.5 * SQ(  vel_x                ) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[2]  = weight[2]  * den * (1 + 3 * (- vel_x                ) + 4.5 * SQ(- vel_x                ) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[3]  = weight[3]  * den * (1 + 3 * (          vel_y        ) + 4.5 * SQ(          vel_y        ) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[4]  = weight[4]  * den * (1 + 3 * (        - vel_y        ) + 4.5 * SQ(        - vel_y        ) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[5]  = weight[5]  * den * (1 + 3 * (                  vel_z) + 4.5 * SQ(                  vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[6]  = weight[6]  * den * (1 + 3 * (                - vel_z) + 4.5 * SQ(                - vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[7]  = weight[7]  * den * (1 + 3 * (  vel_x + vel_y + vel_z) + 4.5 * SQ(  vel_x + vel_y + vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[8]  = weight[8]  * den * (1 + 3 * (- vel_x - vel_y - vel_z) + 4.5 * SQ(- vel_x - vel_y - vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[9]  = weight[9]  * den * (1 + 3 * (  vel_x + vel_y - vel_z) + 4.5 * SQ(  vel_x + vel_y - vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[10] = weight[10] * den * (1 + 3 * (- vel_x - vel_y + vel_z) + 4.5 * SQ(- vel_x - vel_y + vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[11] = weight[11] * den * (1 + 3 * (  vel_x - vel_y + vel_z) + 4.5 * SQ(  vel_x - vel_y + vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[12] = weight[12] * den * (1 + 3 * (- vel_x + vel_y - vel_z) + 4.5 * SQ(- vel_x + vel_y - vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[13] = weight[13] * den * (1 + 3 * (- vel_x + vel_y + vel_z) + 4.5 * SQ(- vel_x + vel_y + vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  pop_eq[14] = weight[14] * den * (1 + 3 * (  vel_x - vel_y - vel_z) + 4.5 * SQ(  vel_x - vel_y - vel_z) - 1.5 * (SQ(vel_x) + SQ(vel_y) + SQ(vel_z)));
  return;
}

// **********************
// PERFORM LBM OPERATIONS
// **********************

void LBM() {

  // The code uses old and new populations which are swapped at the beginning of each time step.
  // This way, the old populations are not overwritten during propagation.

  double ****swap_temp = pop_old;
  pop_old = pop;
  pop = swap_temp;
  // cout << ****swap_temp << endl;

  // The lattice Boltzmann equation is solved in the following.
  // The algorithm includes
  // - computation of the lattice force
  // - combined collision and propagation (push)
  // - half-way bounce-back (the outermost nodes are solid nodes)

  // compute force terms
  force();

  for(int X = 1; X < Nx - 1; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      for(int Z = 1; Z < Nz - 1; ++Z) {
        // Compute the force-shifted velocity.
        const double vel_x = velocity_x[X][Y][Z] + 0.5 * force_x[X][Y][Z] / density[X][Y][Z];
        const double vel_y = velocity_y[X][Y][Z] + 0.5 * force_y[X][Y][Z] / density[X][Y][Z];
        const double vel_z = velocity_z[X][Y][Z] + 0.5 * force_z[X][Y][Z] / density[X][Y][Z];

        // Compute lattice force (Guo's forcing). equation (6.14) with factors in (6.25) in LBM book.
        force_latt[0]  = (1 - 0.5 * omega) * weight[0]  * (3 * ((   - vel_x) * (force_x[X][Y][Z]) + (   - vel_y) * force_y[X][Y][Z] + (   - vel_z) * force_z[X][Y][Z]));
        force_latt[1]  = (1 - 0.5 * omega) * weight[1]  * (3 * (( 1 - vel_x) * (force_x[X][Y][Z]) + (   - vel_y) * force_y[X][Y][Z] + (   - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x) * force_x[X][Y][Z]);
        force_latt[2]  = (1 - 0.5 * omega) * weight[2]  * (3 * ((-1 - vel_x) * (force_x[X][Y][Z]) + (   - vel_y) * force_y[X][Y][Z] + (   - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x) * force_x[X][Y][Z]);
        force_latt[3]  = (1 - 0.5 * omega) * weight[3]  * (3 * ((   - vel_x) * (force_x[X][Y][Z]) + ( 1 - vel_y) * force_y[X][Y][Z] + (   - vel_z) * force_z[X][Y][Z]) + 9 * (vel_y) * force_y[X][Y][Z]);
        force_latt[4]  = (1 - 0.5 * omega) * weight[4]  * (3 * ((   - vel_x) * (force_x[X][Y][Z]) + (-1 - vel_y) * force_y[X][Y][Z] + (   - vel_z) * force_z[X][Y][Z]) + 9 * (vel_y) * force_y[X][Y][Z]);
        force_latt[5]  = (1 - 0.5 * omega) * weight[5]  * (3 * ((   - vel_x) * (force_x[X][Y][Z]) + (   - vel_y) * force_y[X][Y][Z] + ( 1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_z) * force_z[X][Y][Z]);
        force_latt[6]  = (1 - 0.5 * omega) * weight[6]  * (3 * ((   - vel_x) * (force_x[X][Y][Z]) + (   - vel_y) * force_y[X][Y][Z] + (-1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_z) * force_z[X][Y][Z]);
        force_latt[7]  = (1 - 0.5 * omega) * weight[7]  * (3 * (( 1 - vel_x) * (force_x[X][Y][Z]) + ( 1 - vel_y) * force_y[X][Y][Z] + ( 1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x + vel_y + vel_z) * (force_x[X][Y][Z] + force_y[X][Y][Z] + force_z[X][Y][Z]));
        force_latt[8]  = (1 - 0.5 * omega) * weight[8]  * (3 * ((-1 - vel_x) * (force_x[X][Y][Z]) + (-1 - vel_y) * force_y[X][Y][Z] + (-1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x + vel_y + vel_z) * (force_x[X][Y][Z] + force_y[X][Y][Z] + force_z[X][Y][Z]));
        force_latt[9]  = (1 - 0.5 * omega) * weight[9]  * (3 * (( 1 - vel_x) * (force_x[X][Y][Z]) + ( 1 - vel_y) * force_y[X][Y][Z] + (-1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x + vel_y - vel_z) * (force_x[X][Y][Z] + force_y[X][Y][Z] - force_z[X][Y][Z]));
        force_latt[10] = (1 - 0.5 * omega) * weight[10] * (3 * ((-1 - vel_x) * (force_x[X][Y][Z]) + (-1 - vel_y) * force_y[X][Y][Z] + ( 1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x + vel_y - vel_z) * (force_x[X][Y][Z] + force_y[X][Y][Z] - force_z[X][Y][Z]));
        force_latt[11] = (1 - 0.5 * omega) * weight[11] * (3 * (( 1 - vel_x) * (force_x[X][Y][Z]) + (-1 - vel_y) * force_y[X][Y][Z] + ( 1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x - vel_y + vel_z) * (force_x[X][Y][Z] - force_y[X][Y][Z] + force_z[X][Y][Z]));
        force_latt[12] = (1 - 0.5 * omega) * weight[12] * (3 * ((-1 - vel_x) * (force_x[X][Y][Z]) + ( 1 - vel_y) * force_y[X][Y][Z] + (-1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x - vel_y + vel_z) * (force_x[X][Y][Z] - force_y[X][Y][Z] + force_z[X][Y][Z]));
        force_latt[13] = (1 - 0.5 * omega) * weight[13] * (3 * ((-1 - vel_x) * (force_x[X][Y][Z]) + ( 1 - vel_y) * force_y[X][Y][Z] + ( 1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x - vel_y - vel_z) * (force_x[X][Y][Z] - force_y[X][Y][Z] - force_z[X][Y][Z]));
        force_latt[14] = (1 - 0.5 * omega) * weight[14] * (3 * (( 1 - vel_x) * (force_x[X][Y][Z]) + (-1 - vel_y) * force_y[X][Y][Z] + (-1 - vel_z) * force_z[X][Y][Z]) + 9 * (vel_x - vel_y - vel_z) * (force_x[X][Y][Z] - force_y[X][Y][Z] - force_z[X][Y][Z]));

        // Compute equilibrium populations.
        equilibrium(density[X][Y][Z], vel_x, vel_y, vel_z); 

        // This is the lattice Boltzmann equation (combined collision and propagation) including external forcing.
        // equation (6.25) in the LBM book, thus relaxation time defined as implicit time scheme and incorporating forcing term.
        pop[ 0][X]    [Y]    [Z]     = pop_old[ 0][X][Y][Z] * (1. - omega) + pop_eq[ 0] * omega + force_latt[ 0];
        pop[ 1][X + 1][Y]    [Z]     = pop_old[ 1][X][Y][Z] * (1. - omega) + pop_eq[ 1] * omega + force_latt[ 1];
        pop[ 2][X - 1][Y]    [Z]     = pop_old[ 2][X][Y][Z] * (1. - omega) + pop_eq[ 2] * omega + force_latt[ 2];
        pop[ 3][X]    [Y + 1][Z]     = pop_old[ 3][X][Y][Z] * (1. - omega) + pop_eq[ 3] * omega + force_latt[ 3];
        pop[ 4][X]    [Y - 1][Z]     = pop_old[ 4][X][Y][Z] * (1. - omega) + pop_eq[ 4] * omega + force_latt[ 4];
        pop[ 5][X]    [Y]    [Z + 1] = pop_old[ 5][X][Y][Z] * (1. - omega) + pop_eq[ 5] * omega + force_latt[ 5];
        pop[ 6][X]    [Y]    [Z - 1] = pop_old[ 6][X][Y][Z] * (1. - omega) + pop_eq[ 6] * omega + force_latt[ 6];
        // cout << "lattice boltzmann equation for old population [7] " << pop_old[7][X][Y][Z] << " with " << X << Y << Z << endl;  
        // The debugger says there is a EXC_BAD_ACCESS error here in this line of code, but I have no idea what's causing it
        pop[ 7][X + 1][Y + 1][Z + 1] = pop_old[ 7][X][Y][Z] * (1. - omega) + pop_eq[ 7] * omega + force_latt[ 7];
        pop[ 8][X - 1][Y - 1][Z - 1] = pop_old[ 8][X][Y][Z] * (1. - omega) + pop_eq[ 8] * omega + force_latt[ 8];
        pop[ 9][X + 1][Y + 1][Z - 1] = pop_old[ 9][X][Y][Z] * (1. - omega) + pop_eq[ 9] * omega + force_latt[ 9];
        pop[10][X - 1][Y - 1][Z + 1] = pop_old[10][X][Y][Z] * (1. - omega) + pop_eq[10] * omega + force_latt[10];
        pop[11][X + 1][Y - 1][Z + 1] = pop_old[11][X][Y][Z] * (1. - omega) + pop_eq[11] * omega + force_latt[11];
        pop[12][X - 1][Y + 1][Z - 1] = pop_old[12][X][Y][Z] * (1. - omega) + pop_eq[12] * omega + force_latt[12];
        pop[13][X - 1][Y + 1][Z + 1] = pop_old[13][X][Y][Z] * (1. - omega) + pop_eq[13] * omega + force_latt[13];
        pop[14][X + 1][Y - 1][Z - 1] = pop_old[14][X][Y][Z] * (1. - omega) + pop_eq[14] * omega + force_latt[14];
        // cout << "finish LBM for " << X << Y << Z << endl; 
      }
    }
  }

  // Bounce-back
  // Due to the presence of the rigid walls at y = 0, y = Ny - 1, x = 0, x = Nx - 1, z = 0, z = Nz - 1, the populations have to be bounced back.
  // Ladd's momentum correction term is included for moving walls (wall velocity parallel to x-axis).
  // Periodicity of the lattice in x-direction is taken into account via the %-operator.
  for(int X = 1; X < Nx - 1; ++X) {
    for(int Z = 1; Z < Nz - 1; ++Z) {

      // Front wall (y = 0)
      pop[3] [X][1][Z] = pop[4] [X][0][Z];
      pop[7] [X][1][Z] = pop[8] [X - 1][0][Z - 1];
      pop[9] [X][1][Z] = pop[10][X - 1][0][Z + 1];
      pop[12][X][1][Z] = pop[11][X + 1][0][Z + 1];
      pop[13][X][1][Z] = pop[14][X + 1][0][Z - 1];

      // Back wall (y = Ny - 1)
      pop[4] [X][Ny-2][Z] = pop[3] [X][Ny-1][Z];
      pop[8] [X][Ny-2][Z] = pop[7] [X + 1][Ny-1][Z + 1] - 6 * weight[ 7] * density[X][Ny-1][Z] * (   vel_back);
      pop[10][X][Ny-2][Z] = pop[9] [X + 1][Ny-1][Z - 1] - 6 * weight[ 9] * density[X][Ny-1][Z] * (   vel_back);
      pop[11][X][Ny-2][Z] = pop[12][X - 1][Ny-1][Z - 1] - 6 * weight[12] * density[X][Ny-1][Z] * ( - vel_back);
      pop[14][X][Ny-2][Z] = pop[13][X - 1][Ny-1][Z + 1] - 6 * weight[13] * density[X][Ny-1][Z] * ( - vel_back);
    }
  }

  for(int Y = 1; Y < Ny - 1; ++Y) {
    for(int Z = 1; Z < Nz - 1; ++Z) {

      // Left wall (x = 0)
      pop[1] [1][Y][Z] = pop[2] [0][Y][Z];
      pop[7] [1][Y][Z] = pop[8] [0][Y - 1][Z - 1];
      pop[9] [1][Y][Z] = pop[10][0][Y - 1][Z + 1];
      pop[11][1][Y][Z] = pop[12][0][Y + 1][Z - 1];
      pop[14][1][Y][Z] = pop[13][0][Y + 1][Z + 1];

      // Right wall (x = Nx - 1)
      pop[2] [Nx-2][Y][Z] = pop[1] [Nx-1][Y][Z];
      pop[8] [Nx-2][Y][Z] = pop[7] [Nx-1][Y + 1][Z + 1];
      pop[10][Nx-2][Y][Z] = pop[9] [Nx-1][Y + 1][Z - 1];
      pop[12][Nx-2][Y][Z] = pop[11][Nx-1][Y - 1][Z + 1];
      pop[13][Nx-2][Y][Z] = pop[14][Nx-1][Y - 1][Z - 1];
    }
  }

  for(int X = 1; X < Nx - 1; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      // Top wall (z = 0)
      pop[6] [X][Y][1] = pop[5] [X][Y][0];
      pop[8] [X][Y][1] = pop[7] [X + 1][Y + 1][0];
      pop[9] [X][Y][1] = pop[10][X - 1][Y + 1][0];
      pop[12][X][Y][1] = pop[11][X + 1][Y - 1][0];
      pop[14][X][Y][1] = pop[13][X - 1][Y + 1][0];

      // Bottom wall (z = Nz - 1)
      pop[5] [X][Y][Nz-2] = pop[6] [X][Y][Nz-1];
      pop[7] [X][Y][Nz-2] = pop[8] [X - 1][Y - 1][Nz-1];
      pop[10][X][Y][Nz-2] = pop[9] [X + 1][Y + 1][Nz-1];
      pop[11][X][Y][Nz-2] = pop[12][X - 1][Y + 1][Nz-1];
      pop[13][X][Y][Nz-2] = pop[14][X + 1][Y - 1][Nz-1];
    }
  }

  // Edges Bounce-Back
  for(int X = 1; X < Nx - 1; ++X) {
    
    // y = 0, z = 0
    pop[3] [X][1][1] = pop[4] [X][0][0];
    pop[5] [X][1][1] = pop[6] [X][0][0];
    pop[7] [X][1][1] = pop[8] [X - 1][0][0];
    pop[13][X][1][1] = pop[14][X + 1][0][0];

    // y = 0, z = Nz - 1
    pop[3] [X][1][Nz-2] = pop[4] [X][0][Nz-1];
    pop[6] [X][1][Nz-2] = pop[5] [X][0][Nz-1];
    pop[9] [X][1][Nz-2] = pop[10][X - 1][0][Nz-1];
    pop[12][X][1][Nz-2] = pop[11][X + 1][0][Nz-1];

    // y = Ny - 1, z = 0, moving boundary
    pop[4] [X][Ny-2][1] = pop[ 3][X][Ny-1][0];
    pop[5] [X][Ny-2][1] = pop[ 6][X][Ny-1][0];
    pop[10][X][Ny-2][1] = pop[ 9][X + 1][Ny-1][0] - 6 * weight[ 9] * density[X][Ny-1][0] * vel_back;
    pop[11][X][Ny-2][1] = pop[12][X - 1][Ny-1][0] + 6 * weight[12] * density[X][Ny-1][0] * vel_back;

    // y = Ny - 1, z = Nz - 1, moving boundary
    pop[4] [X][Ny-2][Nz-2] = pop[3] [X][Ny-1][Nz-1];
    pop[6] [X][Ny-2][Nz-2] = pop[5] [X][Ny-1][Nz-1];
    pop[8] [X][Ny-2][Nz-2] = pop[7] [X + 1][Ny-1][Nz-1] - 6 * weight[ 7] * density[X][Ny-1][Nz-1] * vel_back;
    pop[14][X][Ny-2][Nz-2] = pop[13][X - 1][Ny-1][Nz-1] + 6 * weight[13] * density[X][Ny-1][Nz-1] * vel_back;
    
  }

  for(int Y = 1; Y < Ny - 1; ++Y) {

    // x = 0, z = 0
    pop[3] [1][Y][1] = pop[4] [0][Y][0];
    pop[5] [1][Y][1] = pop[6] [0][Y][0];
    pop[7] [1][Y][1] = pop[8] [0][Y - 1][0];
    pop[13][1][Y][1] = pop[14][0][Y + 1][0];

    // x = 0, z = Nz - 1
    pop[3] [1][Y][1] = pop[4] [0][Y][0];
    pop[5] [1][Y][1] = pop[6] [0][Y][0];
    pop[7] [1][Y][1] = pop[8] [0][Y - 1][0];
    pop[13][1][Y][1] = pop[14][0][Y + 1][0];

    // x = Nx - 1, z = 0
    pop[3] [1][Y][1] = pop[4] [0][Y][0];
    pop[5] [1][Y][1] = pop[6] [0][Y][0];
    pop[7] [1][Y][1] = pop[8] [0][Y - 1][0];
    pop[13][1][Y][1] = pop[14][0][Y + 1][0];

    // x = Nx - 1, z = Nz - 1
    pop[3] [1][Y][1] = pop[4] [0][Y][0];
    pop[5] [1][Y][1] = pop[6] [0][Y][0];
    pop[7] [1][Y][1] = pop[8] [0][Y - 1][0];
    pop[13][1][Y][1] = pop[14][0][Y + 1][0];
  }

  for(int Z = 1; Z < Nz - 1; ++Z) {
    
    // x = 0, y = 0
    pop[1] [1][1][Z] = pop[2] [0][0][Z];
    pop[3] [1][1][Z] = pop[4] [0][0][Z];
    pop[7] [1][1][Z] = pop[8] [0][0][Z - 1];
    pop[9] [1][1][Z] = pop[10][0][0][Z + 1];

    // x = Nx - 1, y = 0
    pop[2] [Nx - 2][1][Z] = pop[1] [Nx - 1][0][Z];
    pop[3] [Nx - 2][1][Z] = pop[4] [Nx - 1][0][Z];
    pop[12][Nx - 2][1][Z] = pop[11][Nx - 1][0][Z + 1];
    pop[13][Nx - 2][1][Z] = pop[14][Nx - 1][0][Z - 1];

    // x = 0, y = Ny - 1, moving boundary
    pop[1] [1][Ny - 2][Z] = pop[ 2][0][Ny - 1][Z];
    pop[4] [1][Ny - 2][Z] = pop[ 3][0][Ny - 1][Z];
    pop[11][1][Ny - 2][Z] = pop[10][0][Ny - 1][Z + 1] - 6 * weight[10] * density[0][Ny - 1][Z] * vel_back;
    pop[14][1][Ny - 2][Z] = pop[13][0][Ny - 1][Z - 1] + 6 * weight[13] * density[0][Ny - 1][Z] * vel_back;

    // x = Nx - 1, y = Ny-1, moving boundary
    pop[ 2][Nx - 2][Ny - 2][Z] = pop[ 1][Nx - 1][Ny - 1][Z];
    pop[ 4][Nx - 2][Ny - 2][Z] = pop[ 3][Nx - 1][Ny - 1][Z];
    pop[ 8][Nx - 2][Ny - 2][Z] = pop[ 7][Nx - 1][Ny - 1][Z + 1] - 6 * weight[7]  * density[Nx - 1][Ny - 1][Z] * vel_back;
    pop[10][Nx - 2][Ny - 2][Z] = pop[11][Nx - 1][Ny - 1][Z - 1] + 6 * weight[11] * density[Nx - 1][Ny - 1][Z] * vel_back;
    
  }

  // Vertices Bounce-Back

  // x = 0, y = 0, z = 0
  pop[ 7][1]     [1]     [1]      = pop[ 8][0]     [0]     [0];

  // x = 0, y = 0, z = Nz - 1
  pop[ 9][1]     [1]     [Nz - 2] = pop[10][0]     [0]     [Nz - 1];

  // x = 0, y = Ny - 1, z = 0
  pop[11][1]     [Ny - 2][1]      = pop[12][0]     [Ny - 1][0]      + 6 * weight[12] * density[0]     [Ny - 1][0]      * vel_back;

  // x = 0, y = Ny - 1, z = Nz - 1
  pop[14][1]     [Ny - 2][Nz - 2] = pop[13][0]     [Ny - 2][Nz - 1] + 6 * weight[13] * density[0]     [Ny - 2][Nz - 1] * vel_back;

  // x = Nx - 1, y = 0, z = 0
  pop[13][Nx - 2][1]     [1]      = pop[14][Nx - 1][0]     [0];

  // x = Nx - 1, y = 0, z = Nz - 1
  pop[12][Nx - 2][1]     [Nz - 2] = pop[11][Nx - 1][0]     [Nz - 1];

  // x = Nx - 1, y = Ny - 1, z = 0
  pop[10][Nx - 2][Ny - 2][1]      = pop[ 9][Nx - 1][Ny - 1][0]      - 6 * weight[ 9] * density[Nx - 1][Ny - 1][0]      * vel_back;

  // x = Nx - 1, y = Ny - 1, z = Nz - 1
  pop[ 8][Nx - 2][Ny - 2][Nz - 2] = pop[ 7][Nx - 1][Ny - 1][Nz - 1] - 6 * weight[ 7] * density[Nx - 1][Ny - 1][Nz - 1] * vel_back;


  // The new fluid density and velocity are obtained from the populations.
  momenta();
  return;
}

// **********************************
// COMPUTE FLUID DENSITY AND VELOCITY
// **********************************

// This function computes the fluid density and velocity from the populations.
// The velocity correction due to body force is *not* included here.

void momenta() {
  for(int X = 1; X < Nx - 1; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      for(int Z = 1; Z < Nz - 1; ++Z) {
        density[X][Y][Z] = pop[0][X][Y][Z] + pop[1][X][Y][Z] + pop[2][X][Y][Z] + pop[3][X][Y][Z] + pop[4][X][Y][Z] + pop[5][X][Y][Z] + pop[6][X][Y][Z] + pop[7][X][Y][Z] + pop[8][X][Y][Z] + pop[9][X][Y][Z] + pop[10][X][Y][Z] + pop[11][X][Y][Z] + pop[12][X][Y][Z] + pop[13][X][Y][Z] + pop[14][X][Y][Z];
        velocity_x[X][Y][Z] = (pop[1][X][Y][Z] - pop[2][X][Y][Z] + pop[7][X][Y][Z] - pop[8][X][Y][Z] + pop[9][X][Y][Z] - pop[10][X][Y][Z] + pop[11][X][Y][Z] - pop[12][X][Y][Z] - pop[13][X][Y][Z] + pop[14][X][Y][Z]) / density[X][Y][Z];
        velocity_y[X][Y][Z] = (pop[3][X][Y][Z] - pop[4][X][Y][Z] + pop[7][X][Y][Z] - pop[8][X][Y][Z] + pop[9][X][Y][Z] - pop[10][X][Y][Z] - pop[11][X][Y][Z] + pop[12][X][Y][Z] + pop[13][X][Y][Z] - pop[14][X][Y][Z]) / density[X][Y][Z];
        velocity_z[X][Y][Z] = (pop[5][X][Y][Z] - pop[6][X][Y][Z] + pop[7][X][Y][Z] - pop[8][X][Y][Z] - pop[9][X][Y][Z] + pop[10][X][Y][Z] + pop[11][X][Y][Z] - pop[12][X][Y][Z] + pop[13][X][Y][Z] - pop[14][X][Y][Z]) / density[X][Y][Z];
      }
    }
  }

  return;
}


// ************************************
// WRITE VELOCITY PROFILE TO ASCII FILE
// ************************************

void write_fluid_profile(int time) {

  // Create filename.
  std::stringstream output_filename;
  output_filename << "data/fluid_t" << time << ".dat";
  std::ofstream output_file;
  output_file.open(output_filename.str().c_str());
  
  // Write header.
  output_file << "X Y Z density vel_x vel_y vel_z Q_xx Q_yy Q_xy Q_yz Q_xz\n";
  
  // Write data.
  for(int X = 0; X < Nx; ++X) {
    for(int Y = 0; Y < Ny; ++Y) {
      for(int Z = 0; Z < Nz; ++Z) {
        output_file << X << " " << Y << " " << Z << " " << density[X][Y][Z] << " " << velocity_x[X][Y][Z] + 0.5 * (force_x[X][Y][Z]) / density[X][Y][Z] << " " << velocity_y[X][Y][Z] + 0.5 * force_y[X][Y][Z] / density[X][Y][Z] << " " << velocity_z[X][Y][Z] + 0.5 * force_z[X][Y][Z] / density[X][Y][Z] << " " << Q[0][X][Y][Z] << " " << Q[1][X][Y][Z] << " " << Q[2][X][Y][Z] << " " << Q[3][X][Y][Z] << " " << Q[4][X][Y][Z] << "\n";
      }
    }
  }

  output_file.close();

  return;
}

