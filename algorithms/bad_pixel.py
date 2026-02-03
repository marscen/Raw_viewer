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
        pattern = params.get("pattern", "Mono/None")
        
        overlays = []
        limit = 2000 # Increased limit
        
        # Check if we should split channels
        if pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            # Per-channel processing
            # 0: R, 1: Gr, 2: Gb, 3: B (conceptually, we just want 4 2x2 offsets)
            
            # Offsets for 2x2 grid: (0,0), (0,1), (1,0), (1,1)
            offsets = [(0,0), (0,1), (1,0), (1,1)]
            
            h, w = image_data.shape
            
            running_count = 0
            
            for dy, dx in offsets:
                # Slicing: start:stop:step
                # Sub-image for this channel
                sub_img = image_data[dy::2, dx::2]
                
                # Detect on sub-image
                bad_mask = self.detect_on_plane(sub_img, threshold)
                
                # Map back to global coordinates
                y_idxs, x_idxs = np.where(bad_mask)
                
                # Global coords: 
                # global_y = sub_y * 2 + dy
                # global_x = sub_x * 2 + dx
                
                # We can iterate and add
                for sy, sx in zip(y_idxs, x_idxs):
                    if running_count >= limit:
                        break
                    
                    gy = sy * 2 + dy
                    gx = sx * 2 + dx
                    
                    overlays.append({
                        "type": "point",
                        "coords": (int(gx), int(gy)),
                        "color": "red"
                    })
                    running_count += 1
                
                if running_count >= limit:
                    break
                    
        else:
            # Mono image - full plane processing
            bad_mask = self.detect_on_plane(image_data, threshold)
            y_idxs, x_idxs = np.where(bad_mask)
            
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
            "image": image_data,
            "overlays": overlays,
            "message": f"Detected {len(overlays)}+ bad pixels."
        }

    def detect_on_plane(self, plane: np.ndarray, threshold: int) -> np.ndarray:
        # 3x3 median filter check using neighbor splicing
        # Ignore borders for simplicity
        
        # Center (crop by 1 all around)
        center = plane[1:-1, 1:-1].astype(np.float32)
        
        # Neighbors
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                # Slice: 1+dy : -1+dy (but need to handle end index properly)
                # simpler: slice from [1+dy : h-1+dy, 1+dx : w-1+dx] if we knew h,w
                # Let's use negative indexing carefuly.
                
                # if dy=-1, slice 0:-2. if dy=1, slice 2:. if dy=0, slice 1:-1
                s_y = slice(1+dy, -1+dy) if (1+dy) < (-1+dy) else slice(1+dy, None if -1+dy == 0 else -1+dy) 
                # Wait, slice logic is tricky with negative indices.
                
                # Explicit start/end
                # We want a view of size (H-2, W-2)
                # Original range [1 : -1] -> indices 1 .. H-2
                # Neighbor at dy,dx relative to (y,x) is at (y+dy, x+dx)
                # So we want indices (1+dy) .. (H-2+dy)
                
                # Python slicing: start : stop
                # stop is exclusive.
                # if slicing [1:-1], we get items 1 through H-2. Length H-2.
                # neighbor slice start: 1+dy. Stop: -1+dy (if -1+dy is 0, use None? No, 0 is start).
                # Actually -1 is the last element. We want up to (but not including) the last-shifted.
                
                # Let's use explicit shape math to be safe.
                h, w = plane.shape
                # Target shape: h-2, w-2
                
                n_slice = plane[1+dy : h-1+dy, 1+dx : w-1+dx]
                neighbors.append(n_slice)
        
        # Stack
        stack = np.array(neighbors)
        # Median
        med_val = np.median(stack, axis=0)
        
        # Difference
        diff = np.abs(center - med_val)
        
        # Mask
        local_mask = diff > threshold
        
        # Pad back to original size
        full_mask = np.zeros_like(plane, dtype=bool)
        full_mask[1:-1, 1:-1] = local_mask
        
        return full_mask
