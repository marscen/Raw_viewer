from abc import ABC, abstractmethod
import numpy as np

class Algorithm(ABC):
    """
    Abstract base class for all image processing algorithms.
    """
    
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the display name of the algorithm."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Returns a short description of the algorithm."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Returns a dictionary defining the parameters.
        Format:
        {
            "param_name": {
                "type": "int" | "float" | "bool" | "list",
                "default": value,
                "min": value, (optional)
                "max": value, (optional)
                "options": [], (optional for list)
                "label": "Display Label"
            }
        }
        """
        pass

    @abstractmethod
    def run(self, image_data: np.ndarray, params: dict):
        """
        Runs the algorithm on the given image data.
        ARGS:
            image_data: 2D numpy array (H, W) for raw data.
            params: Dictionary of parameter values.
            
        RETURNS:
            dict with keys:
                "image": (optional) modified image data (numpy array)
                "overlays": (optional) list of dicts for visualization
                    [
                        {"type": "point", "coords": (x, y), "color": "red"},
                        {"type": "line", "coords": (x1, y1, x2, y2), "color": "blue"},
                        {"type": "rect", "coords": (x, y, w, h), "color": "green"}
                    ]
                "message": (optional) status message string
        """
        pass
