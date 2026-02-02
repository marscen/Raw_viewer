import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.bad_pixel import BadPixelDetectionAlgorithm

def test_bad_pixel_detection():
    print("Testing Bad Pixel Detection...")
    
    # Create synthetic image 100x100
    h, w = 100, 100
    img = np.full((h, w), 100, dtype=np.uint16)
    
    # Add noise
    noise = np.random.randint(-5, 5, (h, w))
    img = (img + noise).astype(np.uint16)
    
    # Add known bad pixels
    bad_pixels = [
        (10, 10, 4000), # Hot pixel
        (50, 50, 0),    # Dead pixel (might not be detected by simple threshold if background is dark? logic depends)
        (20, 80, 2000)
    ]
    
    for y, x, val in bad_pixels:
        img[y, x] = val
        
    algo = BadPixelDetectionAlgorithm()
    params = algo.get_parameters()
    # Extract defaults
    run_params = {k: v['default'] for k, v in params.items()}
    run_params['threshold'] = 50 # Lower threshold for test
    
    result = algo.run(img, run_params)
    
    overlays = result.get('overlays', [])
    print(f"Detected {len(overlays)} bad pixels.")
    
    detected_coords = []
    for o in overlays:
        if o['type'] == 'point':
            detected_coords.append(o['coords'])
            
    # Check if known bad pixels are detected
    # Note: Algorithm returns (x, y)
    
    expected_detected = 0
    for y, x, val in bad_pixels:
        if (x, y) in detected_coords:
            print(f"PASS: Detected bad pixel at ({x}, {y})")
            expected_detected += 1
        else:
            print(f"FAIL: Missed bad pixel at ({x}, {y})")
            
    if expected_detected == len(bad_pixels):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

if __name__ == "__main__":
    test_bad_pixel_detection()
