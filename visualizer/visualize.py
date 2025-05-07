import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

def load_density_file(filename, grid_size):
    # Read the raw binary file
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    
    # Reshape the 1D array into a 3D grid
    return data.reshape((grid_size, grid_size, grid_size))

def visualize_density_slices(filename, grid_size=128, output_dir='plots'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the density data
    density = load_density_file(filename, grid_size)
    
    # Create a figure with ImageGrid
    fig = plt.figure(figsize=(15, 5))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.1,
                    share_all=True)
    
    # Plot three orthogonal slices through the center
    mid = grid_size // 2
    
    # XY slice (top view)
    im0 = grid[0].imshow(density[:, :, mid], cmap='viridis')
    grid[0].set_title('XY Slice (Top View)')
    
    # XZ slice (front view)
    im1 = grid[1].imshow(density[:, mid, :], cmap='viridis')
    grid[1].set_title('XZ Slice (Front View)')
    
    # YZ slice (side view)
    im2 = grid[2].imshow(density[mid, :, :], cmap='viridis')
    grid[2].set_title('YZ Slice (Side View)')
    
    # Add colorbar
    plt.colorbar(im0, ax=grid.axes_all)
    
    # Get frame number from filename
    frame_num = filename.split('_')[-1].split('.')[0]
    
    plt.suptitle(f'Density Visualization - Frame {frame_num}')
    
    # Save the plot
    output_filename = os.path.join(output_dir, f'slices_{frame_num}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved slices visualization to {output_filename}")

def visualize_volume(filename, grid_size=128, output_dir='plots'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the density data
    density = load_density_file(filename, grid_size)
    
    # Create a figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of points
    x, y, z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size))
    
    # Plot points where density is above a threshold
    threshold = 0.1  # Adjust this threshold as needed
    mask = density > threshold
    
    # Plot the points
    scatter = ax.scatter(x[mask], y[mask], z[mask], c=density[mask], cmap='viridis', alpha=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Get frame number from filename
    frame_num = filename.split('_')[-1].split('.')[0]
    
    plt.title(f'Volume Visualization - Frame {frame_num}')
    
    # Add colorbar
    plt.colorbar(scatter)
    
    # Save the plot
    output_filename = os.path.join(output_dir, f'volume_{frame_num}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved volume visualization to {output_filename}")

def process_all_frames(start_frame=0, end_frame=100, grid_size=128):
    """Process all frames in the range and save visualizations"""
    for frame in range(start_frame, end_frame):
        filename = f"../fluidCuda/build/rendered/density_{frame:04d}.raw"
        if os.path.exists(filename):
            print(f"Processing frame {frame}...")
            visualize_density_slices(filename, grid_size)
            visualize_volume(filename, grid_size)
        else:
            print(f"Warning: File {filename} not found")

if __name__ == "__main__":
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Process all frames from 0 to 99
    process_all_frames(0, 100, 32)
    
    print("\nAll visualizations have been saved to the 'plots' directory")
    print("Files are named as:")
    print("- slices_XXXX.png for 2D slice visualizations")
    print("- volume_XXXX.png for 3D volume visualizations")