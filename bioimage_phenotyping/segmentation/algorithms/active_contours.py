import numpy as np
from ..base import Segmenter
from skimage.segmentation import active_contour

class ActiveContoursSegmenter(Segmenter):
    def __init__(self, snake, alpha=0.01, beta=0.1, w_line=0, w_edge=1, gamma=0.01, max_px_move=1.0, max_iterations=2500, convergence=0.1):
        super().__init__()
        self.snake = snake
        self.alpha = alpha
        self.beta = beta
        self.w_line = w_line
        self.w_edge = w_edge
        self.gamma = gamma
        self.max_px_move = max_px_move
        self.max_iterations = max_iterations
        self.convergence = convergence

    def segment(self, image: np.ndarray) -> np.ndarray:
        return active_contour(
            image,
            self.snake,
            alpha=self.alpha,
            beta=self.beta,
            w_line=self.w_line,
            w_edge=self.w_edge,
            gamma=self.gamma,
            max_px_move=self.max_px_move,
            max_iterations=self.max_iterations,
            convergence=self.convergence
        )
