from dataclasses import dataclass
import io
import math
from typing import Union

import numpy as np
import PIL.Image
from PIL.Image import Image



@dataclass
class ImageState:
    visible_image: Image
    real_image: Image


class ImageValueError(ValueError):
    """
    This exception indicates that something is wrong
    with the image(s) passed as an input.
    """
    pass


SUPPORTED_UPLOAD_FORMATS = ['PNG', 'BMP', 'TIFF', 'JPEG']


def upload_image(filepath: str) -> ImageState:
    image = _upload_image(filepath)
    return ImageState(visible_image=image, real_image=image)


def _upload_image(filepath: str) -> Image:
    try:
        image =  PIL.Image.open(filepath)
    except (OSError, PIL.UnidentifiedImageError):
        raise ValueError(
            'Cannot open the image. '
            f'Ensure that the image format is one of these: {SUPPORTED_UPLOAD_FORMATS}.'
        )

    if image.format not in SUPPORTED_UPLOAD_FORMATS:
        raise ValueError(
            f'Unsupported image format: {image.format}. '
            f'Image format must be one of these: {SUPPORTED_UPLOAD_FORMATS}.'
    )
    return image


def image_filename(image_state: ImageState) -> str:
    return image_state.real_image.filename


def image_to_bytes(image_state: ImageState):
    return _image_to_bytes(image_state.visible_image)


def _image_to_bytes(image: Image) -> bytes:
    bio = io.BytesIO()
    image.save(bio, format='PNG')
    return bio.getvalue()


SAVE_SUPPORTED_MODES = ['L', 'RGB']


def save_image(image_state: ImageState, filepath: str) -> None:
    _save_image(image_state.real_image, filepath)


def _save_image(image: Image, filepath: str) -> None:
    if image.mode not in SAVE_SUPPORTED_MODES:
        raise ImageValueError(
            f'Cannot save image in {image.mode} mode. '
            f'Supported modes are: {SAVE_SUPPORTED_MODES}.'
    )
    image.save(filepath, 'PNG')


MODE_DEPTHS = {
    'L': 8,
    'RGB': 8,
}
PSNR_SUPPORTED_MODES = list(MODE_DEPTHS.keys())


def psnr(image_state1: ImageState, image_state2: ImageState) -> float:
    return _psnr(image_state1.real_image, image_state2.real_image)


def _psnr(image1: Image, image2: Image) -> float:
    for image in (image1, image2):
        if image.mode not in {'L', 'RGB'}:
            raise ImageValueError(
                f'Cannot calculate PSNR for {image.mode} image mode. '
                f'Supported image modes are: {PSNR_SUPPORTED_MODES}.'
            )

    if image1.mode != image2.mode:
        raise ImageValueError(
            'Cannot calculate PSNR for images of different modes. '
            f'Left image has mode {image1.mode}, but right image has mode {image2.mode}.'
        )
    if image1.size != image2.size:
        raise ImageValueError(
            'Cannot calculate PSNR for images of different sizes. '
            f'Left image has size {image1.size}, but right image has size {image2.size}.'
        )

    a1 = np.asarray(image1).astype(int)
    a2 = np.asarray(image2).astype(int)

    if len(a1.shape) == 2:
        height, width = a1.shape
        n_channels = 1
    else:
        height, width, n_channels = a1.shape
    N = height * width * n_channels
    depth = MODE_DEPTHS[image.mode]

    square_error_sum = ((a1 - a2) ** 2).sum()
    if square_error_sum == 0:
        return float('inf')

    max_square_error_sum = N * (2 ** depth - 1) ** 2
    return 10 * math.log10(max_square_error_sum / square_error_sum)


TO_GRAYSCALE_METHODS = ['mean', 'CCIR 601-1']


def to_grayscale(image_state: ImageState, method: str) -> ImageState:
    image = _to_grayscale(image_state.real_image, method=method)
    return ImageState(visible_image=image, real_image=image)


def _to_grayscale(image: Image, method: str) -> Image:
    if image.mode == 'L':
        return image

    if image.mode != 'RGB':
        raise ImageValueError(
            f'Cannot convert image in mode {image.mode} to grayscale. '
            'Image mode must be RGB.'
        )

    if method not in TO_GRAYSCALE_METHODS:
        raise ValueError(
            f'Unknown method {method} for conversion to grayscale.'
    )

    if method == 'mean':
        return _to_grayscale_mean(image)
    return _to_grayscale_ccir601(image)


def _to_grayscale_mean(image: Image) -> Image:
    array = np.asarray(image).astype(int)
    grayscale_array = (array.sum(axis=2) / 3).astype(np.uint8)
    grayscale_image = PIL.Image.fromarray(grayscale_array, mode='L')
    return grayscale_image


def _to_grayscale_ccir601(image: Image) -> Image:
    array = np.asarray(image).astype(int)
    grayscale_array = ((77 * array[:,:,0] + 150 * array[:,:,1] + 29 * array[:,:,2]) / 256).astype(np.uint8)
    grayscale_image = PIL.Image.fromarray(grayscale_array, mode='L')
    return grayscale_image


def rgb_to_ycbcr(image_state: ImageState) -> ImageState:
    image = _rgb_to_ycbcr(image_state.real_image)
    y_as_grayscale_image = _ycbcr_channel_as_grayscale_image(image, channel='Y')
    return ImageState(real_image=image, visible_image=y_as_grayscale_image)


def _rgb_to_ycbcr(image: Image) -> Image:
    if image.mode == 'YCbCr':
        return image

    if image.mode != 'RGB':
        raise ImageValueError(
            'Image has to be in RGB mode to be converted to YCbCr. '
            f'Now it\'s in {image.mode} mode.'
        )

    array = np.asarray(image).astype(float)
    y = (77 * array[:,:,0] + 150 * array[:,:,1] + 29 * array[:,:,2]) / 256
    cb = 144 * (array[:,:,2] - y) / 256 + 128
    cr = 183 * (array[:,:,0] - y) / 256 + 128

    ycbcr_array = np.around(np.stack((y, cb, cr), axis=2))
    # ycbcr_array[ycbcr_array<np.zeros(ycbcr_array.shape)] = 0
    # ycbcr_array[ycbcr_array>255*np.ones(ycbcr_array.shape)] = 255

    ycbcr_image = PIL.Image.fromarray(ycbcr_array.astype(np.uint8), mode='YCbCr')
    return ycbcr_image


YCBCR_CHANNELS = ['Y', 'Cb', 'Cr']


def ycbcr_channel_as_grayscale_image(image_state: ImageState, channel: str) -> ImageState:
    return ImageState(
        real_image=image_state.real_image,
        visible_image=_ycbcr_channel_as_grayscale_image(
            image_state.real_image,
            channel=channel
        )
    )


def _ycbcr_channel_as_grayscale_image(image: Image, channel: str) -> Image:
    if image.mode != 'YCbCr':
        raise ImageValueError(
            'Image has to be in YCbCr mode to get channels as grayscale images. '
            f'Now it\'s in {image.mode} mode.'
    )
    if channel not in YCBCR_CHANNELS:
        raise ImageValueError(
            f'Wrong YCbCr channel {channel}. '
            f'Must be one of these: {YCBCR_CHANNELS}'
    )

    if channel == 'Y':
        component = 0
    elif channel == 'Cb':
        component = 1
    elif channel == 'Cr':
        component = 2

    array = np.asarray(image)
    return _array_as_grayscale_image(array[:,:,component])


def _array_as_grayscale_image(array: np.ndarray):
    return PIL.Image.fromarray(array, mode='L')


def ycbcr_to_rgb(image_state: ImageState) -> ImageState:
    image = _ycbcr_to_rgb(image_state.real_image)
    return ImageState(real_image=image, visible_image=image)


def _ycbcr_to_rgb(image: Image) -> Image:
    if image.mode == 'RGB':
        return image

    if image.mode != 'YCbCr':
        raise ImageValueError(
            'Image has to be in YCbCr mode to be converted to RGB. '
            f'Now it\'s in {image.mode} mode.'
        )

    array = np.asarray(image).astype(float)
    height, width, _ = array.shape
    shift_array = np.stack((
        np.zeros((height, width)),
        -128*np.ones((height, width)),
        -128*np.ones((height, width)),
    ), axis=2)
    shifted_array = array + shift_array

    r = shifted_array[:,:,0] + (256/183) * shifted_array[:,:,2]
    g = shifted_array[:,:,0] - (5329/15481) * shifted_array[:,:,1] - (11103/15481) * shifted_array[:,:,2]
    b = shifted_array[:,:,0] + (256/144) * shifted_array[:,:,1]

    rgb_array = np.around(np.stack((r, g, b), axis=2))
    rgb_array[rgb_array<np.zeros(rgb_array.shape)] = 0
    rgb_array[rgb_array>255*np.ones(rgb_array.shape)] = 255

    rgb_image = PIL.Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')
    return rgb_image


QUANTIZE_SUPPORTED_MODES = ['RGB', 'YCbCr']
MAX_CHANNEL_DEPTH = 8
ALLOWED_CHANNEL_DEPTHS = list(range(9))


def quantize(
    image_state:
    ImageState,
    channel1_depth: Union[str, int],
    channel2_depth: Union[str, int],
    channel3_depth: Union[str, int]
) -> ImageState:
    image = _quantize(
        image_state.real_image,
        channel1_depth=channel1_depth,
        channel2_depth=channel2_depth,
        channel3_depth=channel3_depth
    )
    if image.mode == 'YCbCr':
        y_as_grayscale_image = _ycbcr_channel_as_grayscale_image(image, channel='Y')
        return ImageState(visible_image=y_as_grayscale_image, real_image=image)

    return ImageState(visible_image=image, real_image=image)


def _quantize(
    image: Image,
    channel1_depth: Union[str, int],
    channel2_depth: Union[str, int],
    channel3_depth: Union[str, int]
) -> Image:
    if image.mode not in QUANTIZE_SUPPORTED_MODES:
        raise ImageValueError(
            f'Image must be in one of these modes to be quantized: {QUANTIZE_SUPPORTED_MODES}. '
            f'Now it\'s in the {image.mode} mode'
        )

    array = np.asarray(image).astype(np.uint8)
    channel1 = _quantize_channel(array[:,:,0], channel1_depth)
    channel2 = _quantize_channel(array[:,:,1], channel2_depth)
    channel3 = _quantize_channel(array[:,:,2], channel3_depth)

    array = np.stack((channel1, channel2, channel3), axis=2)
 
    return PIL.Image.fromarray(array, mode=image.mode)



def _quantize_channel(channel: np.ndarray, channel_depth: Union[str, int]):
    try:
        channel_depth = int(channel_depth)
    except ValueError:
        raise ValueError('Channel depth must be an integer value between 0 and 8.')

    if channel_depth not in ALLOWED_CHANNEL_DEPTHS:
        raise ValueError('Channel depth must be an integer value between 0 and 8')

    if channel_depth == MAX_CHANNEL_DEPTH:
        return channel

    bits_to_drop = 8 - channel_depth
    shift = 2 ** (bits_to_drop - 1)
    # mask = 1 << bits_to_drop
    return (channel >> bits_to_drop << bits_to_drop) + shift