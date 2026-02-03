
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from algorithms.bad_pixel import BadPixelDetectionAlgorithm

def test_bad_pixel_detection_bayer():
    print("Testing Bad Pixel Detection logic...")
    
    # Create a 100x100 BGGR image
    # BGGR:
    # B G
    # G R
    
    width = 100
    height = 100
    img = np.zeros((height, width), dtype=np.uint16)
    
    # Fill with flat values for each channel to simulate "smooth" area
    # R constant 1000, G constant 2000, B constant 3000
    
    # B: (0,0), (0,2)... (even row, even col)
    # G1: (0,1), (0,3)... (even row, odd col)
    # G2: (1,0), (1,2)... (odd row, even col)
    # R: (1,1), (1,3)... (odd row, odd col)
    
    # BGGR Layout
    img[0::2, 0::2] = 3000 # B
    img[0::2, 1::2] = 2000 # G1
    img[1::2, 0::2] = 2000 # G2
    img[1::2, 1::2] = 1000 # R
    
    # Inject a bad pixel in R channel
    # Location (5, 5) is indeed R (Odd row, Odd col) in BGGR
    bad_y, bad_x = 5, 5
    img[bad_y, bad_x] = 5000 # Spike
    
    algo = BadPixelDetectionAlgorithm()
    params = {
        "threshold": 500,
        "pattern": "BGGR"
    }
    
    result = algo.run(img, params)
    overlays = result['overlays']
    
    detected = False
    for ov in overlays:
        if ov['coords'] == (bad_x, bad_y):
            detected = True
            print(f"Success: Detected bad pixel at {bad_x}, {bad_y}")
            break
            
    if not detected:
        print("FAILURE: Did not detect bad pixel.")
        print("Overlays found:", overlays)
        sys.exit(1)
        
    # Check false positive
    # Neighbors of (5,5) are G pixels (2000)
    # If we didn't split channels, (5,5) [R=5000] vs neighbors [G=2000] might trigger if threshold < 3000
    # But wait, original R=1000. 
    # Let's test non-bad pixel area.
    # Pixel at (4,4) is B=3000. Neighbors are G=2000. Diff = 1000.
    # If threshold=500 and NO split, (4,4) would look like a bad pixel.
    # With split, (4,4) compares to (2,2), (2,4)... which are all B=3000. Diff=0.
    
    # Start fresh for false positive test
    img2 = np.zeros((height, width), dtype=np.uint16)
    img2[0::2, 0::2] = 3000 # B
    img2[0::2, 1::2] = 2000 # G1
    img2[1::2, 0::2] = 2000 # G2
    img2[1::2, 1::2] = 1000 # R
    
    # No bad pixels added. 
    # Global median check would fail? 
    # Local check without split:
    # Center (4,4) = 3000. Neighbors (3,4)=G=2000, (4,3)=G=2000, (4,5)=G=2000, (5,4)=G=2000...
    # Median of neighbors approx 2000.
    # Diff = 1000.
    # Threshold = 500.
    # Should be detected as FALSE POSITIVE if logic is wrong.
    
    result2 = algo.run(img2, params)
    overlays2 = result2['overlays']
    
    if len(overlays2) > 0:
        print(f"FAILURE: False positives detected! Count: {len(overlays2)}")
        print("First few:", overlays2[:5])
        sys.exit(1)
    else:
        print("Success: No false positives in clean bayer pattern.")

if __name__ == "__main__":
    test_bad_pixel_detection_bayer()
