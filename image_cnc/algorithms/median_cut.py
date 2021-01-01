from dataclasses import dataclass
from typing import Iterable
import heapq

import numpy as np
from scipy.cluster.vq import vq
import PIL.Image


CHANNELS_NUM = 3


@dataclass(eq=True)
class VBox:
    pixels: np.ndarray
    longest_axis: int
    longest_axis_len: int

    def __init__(self, pixels: np.ndarray):
        dimensions = []
        for c in range(CHANNELS_NUM):
            channel = pixels[:, c]
            dimensions.append(channel.max() - channel.min())
        
        longest_axis = np.argmax(np.asarray(dimensions))
        longest_axis_len = np.max(np.asarray(dimensions))
        pixels = pixels[pixels[:,longest_axis].argsort()]

        self.pixels = pixels
        self.longest_axis = longest_axis
        self.longest_axis_len = longest_axis_len


    @staticmethod
    def from_array(array: np.ndarray):
        pixels = array.reshape(-1, CHANNELS_NUM)
        return VBox(pixels)

    @property
    def size(self):
        return self.pixels.shape[0]


    def __lt__(self, other):
        # Reverse order since heapq doesn't support it directly
        return self.size >= other.size

    
    def centroid(self):
        return self.pixels.sum(axis=0) // self.size



def median_cut(array: np.ndarray, k: int) -> np.ndarray:
    """
    Return `k` centroids found by the Median Cut algorithm.
    """
    vboxes = []
    heapq.heappush(vboxes, VBox.from_array(array))
    for _ in range(k-1):
        vbox = heapq.heappop(vboxes)
        vbox_1, vbox_2 = _split_vbox(vbox)
        heapq.heappush(vboxes, vbox_1)
        heapq.heappush(vboxes, vbox_2)

    centroids = np.asarray([vbox.centroid() for vbox in vboxes]).astype(float)
    return centroids


def _split_vbox(vbox: VBox):
    return VBox(vbox.pixels[:vbox.size//2]), VBox(vbox.pixels[vbox.size//2:])
