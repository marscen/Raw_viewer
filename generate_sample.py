import numpy as np
import os

def generate_sample_raw(width, height, bit_depth, filename):
    max_val = (2 ** bit_depth) - 1
    
    # Create a gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    gradient = (xv * yv * max_val).astype(np.uint16)
    
    # Add some "noise" or pattern to simulate Bayer
    # Just simple grid pattern
    noise = (np.random.rand(height, width) * (max_val * 0.1)).astype(np.uint16)
    
    # Combine (clamp to max_val)
    raw_data = np.clip(gradient + noise, 0, max_val).astype(np.uint16)
    
    # Draw a box in the center
    cx, cy = width // 2, height // 2
    raw_data[cy-50:cy+50, cx-50:cx+50] = max_val // 2
    
    # Save
    raw_data.tofile(filename)
    print(f"Generated {filename}: {width}x{height}, {bit_depth}-bit")

if __name__ == "__main__":
    generate_sample_raw(1920, 1080, 10, "sample_1920x1080_10bit.raw")
