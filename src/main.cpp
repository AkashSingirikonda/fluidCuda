#include "fluid_simulation.h"
#include <iostream>

int main() {
    // Create a 64x64x64 fluid simulation
    int w = 3, h = 3, d = 3;
    FluidSimulation sim(w, h, d);
    
    // Initialize the simulation
    sim.initialize();
    
    // Add velocity
    sim.addVelocity(1, 1, 1, 1, 1, 1);

    // Add density
    sim.addDensity(1, 1, 1, 0.5f);

    // Get data back to host
    std::vector<float> density_field = sim.getDensityFieldHost();
    std::vector<float> velocity_x = sim.getVelocityFieldHost(0);  // x component
    std::vector<float> velocity_y = sim.getVelocityFieldHost(1);  // y component
    std::vector<float> velocity_z = sim.getVelocityFieldHost(2);  // z component

    for (int i = 0; i < w * h * d; i++) {
        std::cout << "Density field: " << density_field[i] << std::endl;
        std::cout << "Velocity x: " << velocity_x[i] << std::endl;
        std::cout << "Velocity y: " << velocity_y[i] << std::endl;
        std::cout << "Velocity z: " << velocity_z[i] << std::endl;
    }
    // // Add some initial velocity and density at the center
    // int cx = w / 2, cy = h / 2, cz = d / 2;
    // sim.addVelocity(cx, cy, cz, 1.0f, 0.0f, 0.0f);  // Add velocity at center
    // sim.addDensity(cx, cy, cz, 1.0f);               // Add density at center
    
    // // Main simulation loop
    // for (int i = 0; i < 100; i++) {
    //     sim.step();
        
    //     // Print progress every 10 steps
    //     if (i % 10 == 0) {
    //         std::cout << "Step " << i << " completed" << std::endl;
    //     }
    // }
    
    return 0;
} 