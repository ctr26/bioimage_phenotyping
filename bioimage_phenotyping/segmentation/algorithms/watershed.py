import numpy as np
from ..base import Segmenter
from skimage.segmentation import watershed


import numpy as np
from skimage import filters, segmentation, morphology
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from typing import Callable

from ...pipeline import Pipe, ImageProcessingPipeline


class WatershedSegmenter(Pipe):
    def __init__(
        self,
        threshold_method: Callable = filters.threshold_otsu,
        distance_transform: Callable = ndi.distance_transform_edt,
        **kwargs
    ):
        super().__init__(self.process, **kwargs)
        self.threshold_method = threshold_method
        self.distance_transform = distance_transform

        # self.pipeline = ImageProcessingPipeline()
        # self.pipeline.add_step(self._threshold_image)
        # self.pipeline.add_step(self._apply_distance_transform)
        # self.pipeline.add_step(self._find_local_maxima)
        # self.pipeline.add_step(self._watershed_segmentation)

    def process(self, image: np.ndarray) -> np.ndarray:
        return self._watershed_segmentation(image)

    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        threshold = self.threshold_method(image)
        return image > threshold

    def _apply_distance_transform(self, binary_image: np.ndarray) -> np.ndarray:
        return self.distance_transform(binary_image)

    def _find_local_maxima(self, image: np.array, distance: np.ndarray) -> np.ndarray:
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        return ndi.label(mask)

    def _apply_binary_threshold(self, image: np.ndarray) -> np.ndarray:
        return image > self.threshold_method(image)

    def _watershed_segmentation(self, image: np.ndarray) -> np.ndarray:
        filtered_image = filters.rank.median(image, footprint=np.ones((4, 4)))
        eroded_image = morphology.erosion(filtered_image, footprint=np.ones((3, 3)))
        binary_image = self._apply_binary_threshold(eroded_image)
        distance = self._apply_distance_transform(binary_image)
        markers, _ = self._find_local_maxima(binary_image, distance)
        return segmentation.watershed(-1 * distance, markers, mask=binary_image)
