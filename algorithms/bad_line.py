from .base import Algorithm
import numpy as np

class BadLineDetectionAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "Bad Line Detection"

    @property
    def description(self) -> str:
        return "Detects bad rows or columns by analyzing line averages."

    def get_parameters(self) -> dict:
        return {
            "threshold": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "label": "Threshold"
            },
            "axis": {
                "type": "list",
                "options": ["Rows", "Cols", "Both"],
                "default": "Both",
                "label": "Detect Axis"
            }
        }

    def run(self, image_data: np.ndarray, params: dict):
        threshold = params.get("threshold", 100)
        axis = params.get("axis", "Both")
        pattern = params.get("pattern", "Mono/None")
        
        overlays = []
        
        # Helper to detect lines on a 2D plane
        def detect_lines_on_plane(plane, p_threshold, p_axis_mode):
            # p_axis_mode: "Rows", "Cols", "Both"
            bad_rows = []
            bad_cols = []
            
            h, w = plane.shape
            
            # Detect Rows
            if p_axis_mode in ["Rows", "Both"]:
                # Calculate mean of each row
                row_means = np.mean(plane, axis=1)
                # Median of row means
                global_median = np.median(row_means)
                
                # Simple deviation check
                diff = np.abs(row_means - global_median)
                
                # Find indices
                bad_row_idxs = np.where(diff > p_threshold)[0]
                bad_rows.extend(bad_row_idxs.tolist())
                
            # Detect Cols
            if p_axis_mode in ["Cols", "Both"]:
                # Calculate mean of each col
                col_means = np.mean(plane, axis=0)
                global_median = np.median(col_means)
                
                diff = np.abs(col_means - global_median)
                bad_col_idxs = np.where(diff > p_threshold)[0]
                bad_cols.extend(bad_col_idxs.tolist())
                
            return bad_rows, bad_cols

        # Logic to map plane coords back to global coords
        # Reuse logic from bad_pixel but for lines
        
        if pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            offsets = [(0,0), (0,1), (1,0), (1,1)]
            
            for dy, dx in offsets:
                sub_img = image_data[dy::2, dx::2]
                
                b_rows, b_cols = detect_lines_on_plane(sub_img, threshold, axis)
                
                # Map back
                # Row index r in sub_img corresponds to r*2 + dy in global
                # Col index c in sub_img corresponds to c*2 + dx in global
                
                for r in b_rows:
                    global_r = r * 2 + dy
                    overlays.append({
                        "type": "line",
                        "coords": (0, global_r, image_data.shape[1], global_r), # x1, y1, x2, y2
                        "color": "yellow"
                    })
                    
                for c in b_cols:
                    global_c = c * 2 + dx
                    overlays.append({
                        "type": "line",
                        "coords": (global_c, 0, global_c, image_data.shape[0]),
                        "color": "yellow"
                    })
                    
        else:
            # Mono
            b_rows, b_cols = detect_lines_on_plane(image_data, threshold, axis)
            
            for r in b_rows:
                overlays.append({
                    "type": "line",
                    "coords": (0, r, image_data.shape[1], r),
                    "color": "yellow"
                })
                
            for c in b_cols:
                overlays.append({
                    "type": "line",
                    "coords": (c, 0, c, image_data.shape[0]),
                    "color": "yellow"
                })

        return {
            "image": image_data,
            "overlays": overlays,
            "message": f"Detected {len(overlays)} bad lines."
        }
