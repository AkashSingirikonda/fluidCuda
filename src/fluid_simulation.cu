#include "fluid_simulation.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Constructor
FluidSimulation::FluidSimulation(int width, int height, int depth) {
    params.width = width;
    params.height = height;
    params.depth = depth;
    params.dt = 0.1f;
    params.viscosity = 0.0f;
    params.diffusion = 0.0f;
    
    allocateMemory();
}

// Destructor
FluidSimulation::~FluidSimulation() {
    freeMemory();
}

// Memory allocation
void FluidSimulation::allocateMemory() {
    size_t size = params.width * params.height * params.depth * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_velocityX, size));
    CUDA_CHECK(cudaMalloc(&d_velocityY, size));
    CUDA_CHECK(cudaMalloc(&d_velocityZ, size));
    CUDA_CHECK(cudaMalloc(&d_density, size));
    CUDA_CHECK(cudaMalloc(&d_prevVelocityX, size));
    CUDA_CHECK(cudaMalloc(&d_prevVelocityY, size));
    CUDA_CHECK(cudaMalloc(&d_prevVelocityZ, size));
    CUDA_CHECK(cudaMalloc(&d_prevDensity, size));
    
    resetFields();
}

// Memory cleanup
void FluidSimulation::freeMemory() {
    CUDA_CHECK(cudaFree(d_velocityX));
    CUDA_CHECK(cudaFree(d_velocityY));
    CUDA_CHECK(cudaFree(d_velocityZ));
    CUDA_CHECK(cudaFree(d_density));
    CUDA_CHECK(cudaFree(d_prevVelocityX));
    CUDA_CHECK(cudaFree(d_prevVelocityY));
    CUDA_CHECK(cudaFree(d_prevVelocityZ));
    CUDA_CHECK(cudaFree(d_prevDensity));
}

// Reset all fields to zero
void FluidSimulation::resetFields() {
    size_t size = params.width * params.height * params.depth * sizeof(float);
    CUDA_CHECK(cudaMemset(d_velocityX, 0, size));
    CUDA_CHECK(cudaMemset(d_velocityY, 0, size));
    CUDA_CHECK(cudaMemset(d_velocityZ, 0, size));
    CUDA_CHECK(cudaMemset(d_density, 0, size));
    CUDA_CHECK(cudaMemset(d_prevVelocityX, 0, size));
    CUDA_CHECK(cudaMemset(d_prevVelocityY, 0, size));
    CUDA_CHECK(cudaMemset(d_prevVelocityZ, 0, size));
    CUDA_CHECK(cudaMemset(d_prevDensity, 0, size));
}

// Initialize simulation
void FluidSimulation::initialize() {
    resetFields();
}

// Main simulation step
void FluidSimulation::step() {
    // TODO: Implement the main simulation steps:
    // 1. Velocity step
    // 2. Density step
    // 3. Pressure solve
    // 4. Advection
}

// Add velocity at a point
void FluidSimulation::addVelocity(int x, int y, int z, float vx, float vy, float vz) {
    if (x < 0 || x >= params.width || y < 0 || y >= params.height || z < 0 || z >= params.depth) return;
    int index = (z * params.height + y) * params.width + x;
    CUDA_CHECK(cudaMemcpy(d_prevVelocityX + index, &vx, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prevVelocityY + index, &vy, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prevVelocityZ + index, &vz, sizeof(float), cudaMemcpyHostToDevice));
}

// Add density at a point
void FluidSimulation::addDensity(int x, int y, int z, float amount) {
    if (x < 0 || x >= params.width || y < 0 || y >= params.height || z < 0 || z >= params.depth) return;
    int index = (z * params.height + y) * params.width + x;
    CUDA_CHECK(cudaMemcpy(d_prevDensity + index, &amount, sizeof(float), cudaMemcpyHostToDevice));
}

// Get current density field
float* FluidSimulation::getDensityField() {
    return d_density;
} 