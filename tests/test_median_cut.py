import numpy as np

from image_cnc.algorithms.median_cut import VBox, median_cut


def test_vbox_initialization():
    array = np.asarray([
        [[0,2,4]],
        [[2,10,6]],
    ])

    vbox = VBox.from_array(array)
    assert vbox.longest_axis == 1
    assert vbox.longest_axis_len == 8
    assert vbox.size == 2


def test_one_centroid():
    array = 255 * np.ones((512,512,3))
    centroids = median_cut(array, k=1)
    expected_centroids = np.asarray([255, 255, 255])
    np.array_equal(centroids, expected_centroids)
