from .base import Algorithm
import numpy as np

class BadPixelDetectionAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "Bad Pixel Detection"

    @property
    def description(self) -> str:
        return "Detects hot/dead pixels using a simple threshold deviation from neighbors."

    def get_parameters(self) -> dict:
        return {
            "threshold": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 4095, # Assuming 12-bit max mostly, but can go higher
                "label": "Threshold"
            },
            "visualize_only": {
                "type": "bool",
                "default": True,
                "label": "Visualize Only (No Correction)"
            }
        }

    def run(self, image_data: np.ndarray, params: dict):
        threshold = params.get("threshold", 100)
        # visualize_only = params.get("visualize_only", True)
        
        # Simple detection: Pixel differs from median of 3x3 neighborhood by more than threshold
        # For performance on python, this might be slow on large images without optimization.
        # We'll use openCV or scipy if available, but for now let's try a numpy vector approach or simple loops for demo.
        # Actually, let's just do a very simple check against global mean/std for "Hot" pixels as a placeholder 
        # because 3x3 median filter in pure numpy without scipy.ndimage is verbose to write efficiently.
        
        # Let's assume we can use basic numpy operations. 
        # A simple "hot pixel" check: pixel > mean + N*std or just absolute threshold.
        # Let's do a basic local neighbor check using splicing.
        
        # NOTE: This is a simplified educational implementation. 
        # Real BPC uses median filters.
        
        h, w = image_data.shape
        bad_pixels = []
        
        # Minimal viable demo: Detect pixels > 99% of max dynamic range or simply outliers.
        # Let's try to implement a 3x3 median-ish check using splicing (ignoring borders)
        
        # Center
        center = image_data[1:-1, 1:-1].astype(np.float32)
        
        # Neighbors
        n1 = image_data[0:-2, 0:-2]
        n2 = image_data[0:-2, 1:-1]
        n3 = image_data[0:-2, 2:]
        n4 = image_data[1:-1, 0:-2]
        n5 = image_data[1:-1, 2:]
        n6 = image_data[2:, 0:-2]
        n7 = image_data[2:, 1:-1]
        n8 = image_data[2:, 2:]
        
        # Stack neighbors
        neighbors = np.array([n1, n2, n3, n4, n5, n6, n7, n8])
        med_val = np.median(neighbors, axis=0)
        
        diff = np.abs(center - med_val)
        
        # Make boolean mask
        mask = diff > threshold
        
        # Get coordinates
        # np.where returns tuple of arrays (row_indices, col_indices)
        # Remember we shifted by 1
        y_idxs, x_idxs = np.where(mask)
        
        # Adjust for splicing offset
        y_idxs += 1
        x_idxs += 1
        
        overlays = []
        # Limit the number of overlays to avoid crashing UI if everything is detected
        limit = 1000
        count = 0
        
        for y, x in zip(y_idxs, x_idxs):
            if count >= limit:
                break
            overlays.append({
                "type": "point",
                "coords": (int(x), int(y)),
                "color": "red"
            })
            count += 1
            
        return {
            "image": image_data, # Return original for now (or corrected if implemented)
            "overlays": overlays,
            "message": f"Detected {len(y_idxs)} bad pixels (showing first {limit})."
        }
