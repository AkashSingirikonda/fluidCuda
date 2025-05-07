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
    params.visc = 0.0f;
    params.diff = 0.0f;
    
    allocateMemory();
}

// Destructor
FluidSimulation::~FluidSimulation() {
    freeMemory();
}

// Memory allocation
void FluidSimulation::allocateMemory() {
    size_t size = params.width * params.height * params.depth * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_s, size));
    CUDA_CHECK(cudaMalloc(&d_density, size));

    CUDA_CHECK(cudaMalloc(&d_Vx, size));
    CUDA_CHECK(cudaMalloc(&d_Vy, size));
    CUDA_CHECK(cudaMalloc(&d_Vz, size));
    
    CUDA_CHECK(cudaMalloc(&d_Vx0, size));
    CUDA_CHECK(cudaMalloc(&d_Vy0, size));
    CUDA_CHECK(cudaMalloc(&d_Vz0, size));
    
    resetFields();
}

// Memory cleanup
void FluidSimulation::freeMemory() {
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_density));

    CUDA_CHECK(cudaFree(d_Vx));
    CUDA_CHECK(cudaFree(d_Vy));
    CUDA_CHECK(cudaFree(d_Vz));
    
    CUDA_CHECK(cudaFree(d_Vx0));
    CUDA_CHECK(cudaFree(d_Vy0));
    CUDA_CHECK(cudaFree(d_Vz0));
}

// Reset all fields to zero
void FluidSimulation::resetFields() {
    size_t size = params.width * params.height * params.depth * sizeof(float);

    CUDA_CHECK(cudaMemset(d_s, 0, size));
    CUDA_CHECK(cudaMemset(d_density, 0, size));

    CUDA_CHECK(cudaMemset(d_Vx, 0, size));
    CUDA_CHECK(cudaMemset(d_Vy, 0, size));
    CUDA_CHECK(cudaMemset(d_Vz, 0, size));
    
    CUDA_CHECK(cudaMemset(d_Vx0, 0, size));
    CUDA_CHECK(cudaMemset(d_Vy0, 0, size));
    CUDA_CHECK(cudaMemset(d_Vz0, 0, size));
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

int FluidSimulation::IX(int x, int y, int z) {
    return (z * params.height + y) * params.width + x;
}

// CUDA kernel for adding velocity
__global__ void addVelocityKernel(float* d_Vx, float* d_Vy, float* d_Vz, 
                                int x, int y, int z, float vx, float vy, float vz,
                                int width, int height, int depth) {
    int idx = (z * height + y) * width + x;
    if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
        d_Vx[idx] += vx;
        d_Vy[idx] += vy;
        d_Vz[idx] += vz;
    }
}

// CUDA kernel for adding density
__global__ void addDensityKernel(float* d_density, int x, int y, int z, 
                               float amount, int width, int height, int depth) {
    int idx = (z * height + y) * width + x;
    if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
        d_density[idx] += amount;
    }
}

// Get density field as host vector
std::vector<float> FluidSimulation::getDensityFieldHost() {
    size_t size = params.width * params.height * params.depth;
    std::vector<float> host_data(size);
    CUDA_CHECK(cudaMemcpy(host_data.data(), d_density, size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_data;
}

// Get velocity field as host vector
std::vector<float> FluidSimulation::getVelocityFieldHost(int component) {
    size_t size = params.width * params.height * params.depth;
    std::vector<float> host_data(size);
    
    float* d_field;
    switch(component) {
        case 0: d_field = d_Vx; break;
        case 1: d_field = d_Vy; break;
        case 2: d_field = d_Vz; break;
        default: return std::vector<float>(); // Return empty vector for invalid component
    }
    
    CUDA_CHECK(cudaMemcpy(host_data.data(), d_field, size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_data;
}

// Add velocity at a point (using CUDA kernel)
void FluidSimulation::addVelocity(int x, int y, int z, float vx, float vy, float vz) {
    if (x < 0 || x >= params.width || y < 0 || y >= params.height || z < 0 || z >= params.depth) return;
    
    dim3 block(1);
    dim3 grid(1);
    addVelocityKernel<<<grid, block>>>(d_Vx, d_Vy, d_Vz, x, y, z, vx, vy, vz,
                                      params.width, params.height, params.depth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Add density at a point (using CUDA kernel)
void FluidSimulation::addDensity(int x, int y, int z, float amount) {
    if (x < 0 || x >= params.width || y < 0 || y >= params.height || z < 0 || z >= params.depth) return;
    
    dim3 block(1);
    dim3 grid(1);
    addDensityKernel<<<grid, block>>>(d_density, x, y, z, amount,
                                     params.width, params.height, params.depth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Get current density field
float* FluidSimulation::getDensityField() {
    return d_density;
}
