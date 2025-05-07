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
    float viscosity; // Viscosity coefficient
    float diffusion; // Diffusion coefficient
};

// Fluid simulation class
class FluidSimulation {
public:
    FluidSimulation(int width, int height, int depth);
    ~FluidSimulation();

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

private:
    SimulationParams params;
    
    // Device memory pointers (3D)
    float *d_velocityX;
    float *d_velocityY;
    float *d_velocityZ;
    float *d_density;
    float *d_prevVelocityX;
    float *d_prevVelocityY;
    float *d_prevVelocityZ;
    float *d_prevDensity;

    // Helper functions
    void allocateMemory();
    void freeMemory();
    void resetFields();
}; 