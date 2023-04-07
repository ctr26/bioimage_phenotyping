from functools import singledispatchmethod
from typing import Callable, List, Union, Optional
import dask.array as da

# from dask import delayed, compute
# from dask.distributed import Client
import numpy as np
import PIL


class Pipe:
    def __init__(self, function: Callable, **kwargs):
        self.function = function
        self.kwargs = kwargs

    @singledispatchmethod
    def __call__(self, image):
        raise TypeError("Unsupported image type")

    @__call__.register(PIL.Image.Image)
    def _process_PIL(self, image: PIL.Image.Image) -> PIL.Image:
        return NotImplemented

    @__call__.register(np.ndarray)
    def _process_numpy(self, image: np.ndarray) -> np.ndarray:
        return self.function(image, **self.kwargs)

    @__call__.register(da.Array)
    def _process_dask(self, image: da.Array) -> da.Array:
        return NotImplemented


class ImageProcessingPipeline:
    def __init__(self, steps: List[Pipe] = None):
        self.steps = steps if steps is not None else []
        # self.dask_client = dask_client

    def add_step(self, step: Pipe):
        self.steps.append(step)

    @singledispatchmethod
    def process(self, image):
        raise TypeError("Unsupported image type")

    @process.register(np.ndarray)
    def _process_numpy(self, image: np.ndarray) -> np.ndarray:
        result = image
        for step in self.steps:
            result = step(result)
        return result

    @process.register(PIL.Image.Image)
    def _process_PIL(self, image: PIL.Image.Image) -> PIL.Image:
        return NotImplemented

    @process.register(da.Array)
    def _process_dask(self, image: da.Array) -> da.Array:
        return NotImplemented
