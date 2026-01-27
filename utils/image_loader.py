import numpy as np

def load_raw_image(file_path, width, height, bit_depth):
    """
    Loads a headerless RAW image and converts it to a normalized 8-bit numpy array.
    """
    try:
        # Determine data type based on bit depth
        if bit_depth <= 8:
            dtype = np.uint8
        elif bit_depth <= 16:
            dtype = np.uint16
        else:
            raise ValueError("Unsupported bit depth")
        
        # Read data from file
        raw_data = np.fromfile(file_path, dtype=dtype)
        
        # Check if dimensions match file size
        expected_size = width * height
        if raw_data.size != expected_size:
             # Try to adjust if file is larger (maybe has header or extra data, but we strictly read W*H here? 
             # Or typically raw files are exact. Let's warn or just slice.)
             if raw_data.size > expected_size:
                 raw_data = raw_data[:expected_size]
             else:
                 raise ValueError(f"File size too small for dimensions {width}x{height}")

        # Reshape
        image = raw_data.reshape((height, width))
        
        # Normalize to 8-bit for display purposes
        # If 10, 12, 14, 16 bit, we typically shift or scale. 
        # Simple scaling: (value / max_val) * 255
        max_val = (2 ** bit_depth) - 1
        normalized_image = (image.astype(np.float32) / max_val * 255).astype(np.uint8)
        
        return normalized_image, image # Return both display version and original raw data
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
