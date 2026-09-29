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
        """在原始 DN 数据上运行算法。

        ARGS:
            image_data: (H, W) 原始 DN 数组（raw_io.load_raw 的输出，未归一化）。
            params: 界面参数 + **主窗口注入的上下文**（下划线前缀，避免与界面
                    参数重名）：
                      params['pattern']     当前 CFA pattern（如 'RGGB'/'Mono/None'）
                      params['_bit_depth']  位深，用于算 max_code
                      params['_roi']        (x0, y0, x1, y1) 或 None，勾选"只在 ROI
                                            内运行"时给出
                      params['_reference']  参考帧 DN 数组（与参考帧比较类算法用）

        RETURNS:
            dict:
                "image":     (可选) 结果图（校正后的 DN 数组），主窗口据此更新画面
                "corrected": (可选) True 表示这是一张校正后的新图（可撤销/做差分）
                "overlays":  (可选) 叠加层列表
                    {"type": "point", "coords": (x, y), "kind": "hot"}
                    {"type": "line",  "coords": (x1, y1, x2, y2), "kind": "row"}
                    {"type": "rect",  "coords": (x, y, w, h), "kind": "cluster"}
                    kind 决定颜色：hot/dead/cluster/row/col/sat/shading
                "defects":   (可选) 缺陷清单，元素为 dict：
                    {"type", "x", "y", "channel", "value", "delta", "note"}
                "report":    (可选) 结构化测量结果（供导出/二次处理）
                "message":   (可选) 状态文本（多行也支持）
        """
        pass
