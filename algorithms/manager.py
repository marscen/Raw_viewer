from .base import Algorithm
from .bad_pixel import BadPixelDetectionAlgorithm

class AlgorithmManager:
    """
    Manages available algorithms.
    """
    def __init__(self):
        self.algorithms = {}
        self._register_default_algorithms()

    def _register_default_algorithms(self):
        """Registers built-in algorithms."""
        self.register(BadPixelDetectionAlgorithm())

    def register(self, algorithm: Algorithm):
        """Registers an algorithm instance."""
        if not isinstance(algorithm, Algorithm):
            raise TypeError("Must inherit from Algorithm base class")
        self.algorithms[algorithm.name] = algorithm

    def get_algorithm_names(self):
        """Returns list of registered algorithm names."""
        return list(self.algorithms.keys())

    def get_algorithm(self, name) -> Algorithm:
        """Returns the algorithm instance by name."""
        return self.algorithms.get(name)
