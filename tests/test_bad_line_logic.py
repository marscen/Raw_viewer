
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from algorithms.bad_line import BadLineDetectionAlgorithm

def test_bad_line_detection():
    print("Testing Bad Line Detection logic...")
    
    width = 100
    height = 100
    
    # 1. Test Mono Row Detection
    print("\n[Test 1] Mono Row Detection")
    img = np.full((height, width), 1000, dtype=np.uint16)
    
    # Bad row at index 50, value 2000 (diff 1000)
    img[50, :] = 2000
    
    algo = BadLineDetectionAlgorithm()
    params = {
        "threshold": 500,
        "axis": "Rows",
        "pattern": "Mono/None"
    }
    
    result = algo.run(img, params)
    overlays = result['overlays']
    
    # Expect 1 line overlay
    found = False
    for ov in overlays:
        if ov['type'] == 'line':
            # Coords: x1, y1, x2, y2. Row line: 0, r, w, r
            x1, y1, x2, y2 = ov['coords']
            if y1 == 50 and y2 == 50:
                found = True
                print("Success: Detected bad row at 50")
                break
    
    if not found:
        print("FAILURE: Did not detect bad row.")
        print("Overlays:", overlays)
        sys.exit(1)
        
    if len(overlays) != 1:
         print(f"Warning: Found {len(overlays)} overlays, expected 1.")
         
         
    # 2. Test Bayer Col Detection
    print("\n[Test 2] Bayer Col Detection (BGGR)")
    # Create valid BGGR content
    img2 = np.zeros((height, width), dtype=np.uint16)
    img2[0::2, 0::2] = 3000 # B
    img2[0::2, 1::2] = 2000 # G1
    img2[1::2, 0::2] = 2000 # G2
    img2[1::2, 1::2] = 1000 # R
    
    # Inject bad column in B/G2 columns (Even cols)
    # Col 10 is even. Encounters B (3000) and G2 (2000).
    # Let's make the whole col 10 super bright = 5000.
    # For B channel (0::2, 0::2), this col corresponds to sub-col index 5.
    # B channel baseline 3000. Bad col becomes 5000. Diff 2000.
    # For G2 channel (1::2, 0::2), this col also corresponds to sub-col index 5.
    # G2 channel baseline 2000. Bad col becomes 5000. Diff 3000.
    
    img2[:, 10] = 5000
    
    params2 = {
        "threshold": 500,
        "axis": "Cols",
        "pattern": "BGGR"
    }
    
    result2 = algo.run(img2, params2)
    overlays2 = result2['overlays']
    
    found_col_10 = False
    
    # We expect detection in B channel and G2 channel. Both map back to global col 10.
    # So we might get TWO overlays for the same line? Or list contains duplicates?
    # Our algo appends overlay for each detected sub-line.
    
    count_10 = 0
    for ov in overlays2:
        x1, y1, x2, y2 = ov['coords']
        # Vertical line: x1=x2=col
        if x1 == 10 and x2 == 10:
            count_10 += 1
            
    if count_10 >= 1:
        print(f"Success: Detected bad col at 10 (Count: {count_10})")
    else:
        print("FAILURE: Did not detect bad col at 10.")
        print(overlays2)
        sys.exit(1)

    # 3. False Positive Check
    print("\n[Test 3] False Positive Check")
    # Clean BGGR image.
    # B avg=3000, G1 avg=2000...
    # Without split, row mean would be avg of (3000,2000) = 2500 for even rows
    # and avg of (2000,1000) = 1500 for odd rows.
    # If we do global row median on MIXED data? 
    # Row means: 2500, 1500, 2500, 1500...
    # Median is 2000 (avg of 1500 and 2500) or 2500/1500.
    # Deviations: |2500-2000|=500. |1500-2000|=500.
    # If threshold=400, ALL rows might fail if not split.
    # With split, B channel rows are all 3000. Diff=0.
    
    img3 = np.zeros((height, width), dtype=np.uint16)
    img3[0::2, 0::2] = 3000
    img3[0::2, 1::2] = 2000
    img3[1::2, 0::2] = 2000
    img3[1::2, 1::2] = 1000
    
    params3 = {
        "threshold": 100, # Strict threshold
        "axis": "Rows",
        "pattern": "BGGR"
    }
    
    result3 = algo.run(img3, params3)
    if len(result3['overlays']) == 0:
        print("Success: No false positives.")
    else:
        print(f"FAILURE: False positives. Found {len(result3['overlays'])}")
        sys.exit(1)

if __name__ == "__main__":
    test_bad_line_detection()
