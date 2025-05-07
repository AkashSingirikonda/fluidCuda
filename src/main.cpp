#include "fluid_simulation.h"
#include <iostream>

int main() {
    // Create a 64x64x64 fluid simulation
    int w = 64, h = 64, d = 64;
    FluidSimulation sim(w, h, d);
    
    // Initialize the simulation
    sim.initialize();
    
    // Add some initial velocity and density at the center
    int cx = w / 2, cy = h / 2, cz = d / 2;
    sim.addVelocity(cx, cy, cz, 1.0f, 0.0f, 0.0f);  // Add velocity at center
    sim.addDensity(cx, cy, cz, 1.0f);               // Add density at center
    
    // Main simulation loop
    for (int i = 0; i < 100; i++) {
        sim.step();
        
        // Print progress every 10 steps
        if (i % 10 == 0) {
            std::cout << "Step " << i << " completed" << std::endl;
        }
    }
    
    return 0;
} 