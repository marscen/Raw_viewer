from .base import Algorithm
from .bad_pixel import BadPixelDetectionAlgorithm
from .bad_line import BadLineDetectionAlgorithm
from .shading import ShadingAnalysisAlgorithm
from .clip_check import ClipCheckAlgorithm
from .frame_diff import FrameDiffAlgorithm


class AlgorithmManager:
    """
    算法注册表。界面只通过 get_algorithm_names()/get_algorithm() 访问，
    新增算法只要在 _register_default_algorithms 里加一行。
    """

    def __init__(self):
        self.algorithms = {}
        self._register_default_algorithms()

    def _register_default_algorithms(self):
        """注册内置算法（顺序即界面下拉框顺序）。"""
        for algo in (
            BadPixelDetectionAlgorithm(),      # 坏点
            BadLineDetectionAlgorithm(),       # 坏行/坏列
            ShadingAnalysisAlgorithm(),        # 阴影/暗角
            ClipCheckAlgorithm(),              # 饱和/黑点
            FrameDiffAlgorithm(),              # 与参考帧比较
        ):
            self.register(algo)

    def register(self, algorithm: Algorithm):
        """注册一个算法实例。"""
        if not isinstance(algorithm, Algorithm):
            raise TypeError("Must inherit from Algorithm base class")
        self.algorithms[algorithm.name] = algorithm

    def get_algorithm_names(self):
        """返回已注册算法名列表。"""
        return list(self.algorithms.keys())

    def get_algorithm(self, name) -> Algorithm:
        """按名字取算法实例。"""
        return self.algorithms.get(name)
