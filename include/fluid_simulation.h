#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

// Simulation parameters
struct SimulationParams {
    int width;
    int height;
    int depth;      // Add depth for 3D
    float dt;        // Time step
    float visc; // Viscosity coefficient
    float diff; // Diffusion coefficient
};

// Fluid simulation class
class FluidSimulation {
public:
    FluidSimulation(int width, int height, int depth);
    ~FluidSimulation();

    int IX(int x, int y, int z);
    // Initialize the simulation
    void initialize();

    // Main simulation step
    void step();

    // Add velocity at a point (3D)
    void addVelocity(int x, int y, int z, float vx, float vy, float vz);

    // Add density at a point (3D)
    void addDensity(int x, int y, int z, float amount);

    // Get current density field
    float* getDensityField();

    // Get density field as host vector
    std::vector<float> getDensityFieldHost();

    // Get velocity field as host vector (component: 0=x, 1=y, 2=z)
    std::vector<float> getVelocityFieldHost(int component);

private:
    SimulationParams params;
    
    // Device memory pointers (3D)

    float *d_s;
    float *d_density;
    
    float *d_Vx;
    float *d_Vy;
    float *d_Vz;

    float *d_Vx0;
    float *d_Vy0;
    float *d_Vz0;

    // Helper functions
    void allocateMemory();
    void freeMemory();
    void resetFields();
}; 