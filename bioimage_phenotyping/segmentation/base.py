import numpy as np
from ..pipeline import Pipe, ImageProcessingPipeline
class Segmenter(Pipe):
    def __init__(self):
        self.segment = self.process

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement the 'segment' method.")
