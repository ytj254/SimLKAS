from lanenet.laneNet_class import LaneNet as _LaneNetModel
from .lanenet_detector import LaneDetector


def build_lanenet_detector():
    """Construct LaneNet model and detector wrapper."""
    model = _LaneNetModel()
    return LaneDetector(model)
