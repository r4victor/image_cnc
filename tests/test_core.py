import os.path

import numpy as np
import PIL.Image
import pytest

from image_cnc import core


TEST_DIR = os.path.realpath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(TEST_DIR, 'images')


# fixtures and utility stuff

def get_test_image_path(filename):
    return os.path.join(IMAGE_DIR, filename)


@pytest.fixture(scope='module')
def peppers512rgb_image():
    filepath = get_test_image_path('image_Peppers512rgb.png')
    image = PIL.Image.open(filepath)
    return image


@pytest.fixture(scope='module')
def white512rgb_image():
    white_array = 255 * np.ones((512, 512, 3), dtype=np.uint8)
    white_image = PIL.Image.fromarray(white_array, mode='RGB')
    return white_image


@pytest.fixture(scope='module')
def black512rgb_image():
    black_array = np.zeros((512, 512, 3), dtype=np.uint8)
    black_image = PIL.Image.fromarray(black_array, mode='RGB')
    return black_image


@pytest.fixture(scope='module')
def white512ycbcr_image():
    size = (512, 512)
    white_array = np.stack((
        255 * np.ones(size),
        128 * np.ones(size),
        128 * np.ones(size)
    ), axis=2).astype(np.uint8)
    white_image = PIL.Image.fromarray(white_array, mode='YCbCr')
    return white_image


@pytest.fixture(scope='module')
def black512ycbcr_image():
    size = (512, 512)
    black_array = np.stack((
        np.zeros(size),
        128 * np.ones(size),
        128 * np.ones(size)
    ), axis=2).astype(np.uint8)
    black_image = PIL.Image.fromarray(black_array, mode='YCbCr')
    return black_image


# test upload image

def test_upload_png():
    filepath = os.path.join(IMAGE_DIR, 'image_Peppers512rgb.png')
    _test_upload(filepath)


def test_upload_bmp():
    filepath = os.path.join(IMAGE_DIR, 'boy.bmp')
    _test_upload(filepath)


def test_upload_tif():
    filepath = os.path.join(IMAGE_DIR, 'cameraman.tif')
    _test_upload(filepath)


def test_upload_jpeg():
    filepath = os.path.join(IMAGE_DIR, 'lena.jpg')
    _test_upload(filepath)


def _test_upload(filepath):
    image_state = core.upload_image(filepath)
    assert image_state.real_image is not None
    assert image_state.visible_image is not None


def test_upload_no_such_file():
    filepath = os.path.join(IMAGE_DIR, '32asdfkkpopoksapokp.png')
    with pytest.raises(ValueError):
        core.upload_image(filepath)


def test_upload_not_image():
    filepath = os.path.join(IMAGE_DIR, 'not_image.txt')
    with pytest.raises(ValueError):
        core.upload_image(filepath)


# test psnr

def test_psnr_same_image(peppers512rgb_image):
    image = peppers512rgb_image
    psnr = core._psnr(image, image)
    assert psnr == float('inf')


def test_psnr_black_vs_white_rgb(white512rgb_image, black512rgb_image):
    white_image = white512rgb_image
    black_image = black512rgb_image
    psnr = core._psnr(white_image, black_image)
    assert psnr == 0


def test_psnr_black_vs_white_grayscale():
    white_array = 255 * np.ones((512, 512), dtype=np.uint8)
    white_image = PIL.Image.fromarray(white_array, mode='L')
    black_array = np.zeros((512, 512), dtype=np.uint8)
    black_image = PIL.Image.fromarray(black_array, mode='L')
    psnr = core._psnr(white_image, black_image)
    assert psnr == 0


# test to_grayscale conversion

def test_to_grayscale_mode_mean(peppers512rgb_image):
    image = peppers512rgb_image
    grayscale_image = core._to_grayscale(image, method='mean')
    assert grayscale_image.mode == 'L'


def test_to_grayscale_mode_ccir601(peppers512rgb_image):
    image = peppers512rgb_image
    grayscale_image = core._to_grayscale(image, method='CCIR 601-1')
    assert grayscale_image.mode == 'L'


# test RGB <-> YCbCr conversion

def test_rgb_ycbcr_back_and_forth(peppers512rgb_image):
    image = peppers512rgb_image
    ycbcr_image = core._rgb_to_ycbcr(image)
    rgb_image = core._ycbcr_to_rgb(ycbcr_image)
    # allow pixels to differ no more than by 1
    np.testing.assert_allclose(
        np.asarray(image, dtype=float),
        np.asarray(rgb_image, dtype=float),
        atol=1
    )


def test_white_to_ycbcr(white512rgb_image):
    image = white512rgb_image
    size = (512, 512)
    ycbcr_image = core._rgb_to_ycbcr(image)
    array = np.asarray(ycbcr_image)
    y, cb, cr = array[:,:,0], array[:,:,1], array[:,:,2]
    expected_y = 255 * np.ones(size, dtype=np.uint8)
    expected_cb = expected_cr = 128 * np.ones(size, dtype=np.uint8)
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(cb, expected_cb)
    np.testing.assert_array_equal(cr, expected_cr)


def test_black_to_ycbcr(black512rgb_image):
    image = black512rgb_image
    size = (512, 512)
    ycbcr_image = core._rgb_to_ycbcr(image)
    array = np.asarray(ycbcr_image)
    y, cb, cr = array[:,:,0], array[:,:,1], array[:,:,2]
    expected_y = np.zeros(size, dtype=np.uint8)
    expected_cb = expected_cr = 128 * np.ones(size, dtype=np.uint8)
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(cb, expected_cb)
    np.testing.assert_array_equal(cr, expected_cr)


def test_white_to_rgb(white512ycbcr_image):
    image = white512ycbcr_image
    rgb_image = core._ycbcr_to_rgb(image)
    array = np.asarray(rgb_image)
    r, g, b = array[:,:,0], array[:,:,1], array[:,:,2]
    expected_r = expected_g = expected_b = 255 * np.ones((512, 512), dtype=np.uint8)
    np.testing.assert_array_equal(r, expected_r)
    np.testing.assert_array_equal(g, expected_g)
    np.testing.assert_array_equal(b, expected_b)


def test_black_to_rgb(black512ycbcr_image):
    image = black512ycbcr_image
    rgb_image = core._ycbcr_to_rgb(image)
    array = np.asarray(rgb_image)
    r, g, b = array[:,:,0], array[:,:,1], array[:,:,2]
    expected_r = expected_g = expected_b = np.zeros((512, 512), dtype=np.uint8)
    np.testing.assert_array_equal(r, expected_r)
    np.testing.assert_array_equal(g, expected_g)
    np.testing.assert_array_equal(b, expected_b)
