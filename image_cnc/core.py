from dataclasses import dataclass
import io
import math

import numpy as np
import PIL.Image


@dataclass
class ImageState:
    visible_image: PIL.Image.Image
    real_image: PIL.Image.Image


def upload_image(filepath: str) -> ImageState:
    image = PIL.Image.open(filepath)
    image_state = ImageState(visible_image=image, real_image=image)
    return image_state


def image_filename(image_state: ImageState) -> str:
    return image_state.real_image.filename


def image_to_bytes(image_state: ImageState):
    return _image_to_bytes(image_state.visible_image)


def _image_to_bytes(image: PIL.Image.Image) -> bytes:
    bio = io.BytesIO()
    image.save(bio, format='PNG')
    return bio.getvalue()


def save_image(filepath: str, image_state: ImageState) -> None:
    image = image_state.real_image
    image.save(filepath, 'PNG')


MODE_DEPTHS = {
    'L': 8,
    'RGB': 8,
}
PSNR_SUPPORTED_MODES = set(MODE_DEPTHS.keys())


def psnr(image_state1: ImageState, image_state2: ImageState) -> float:
    return _psnr(image_state1.real_image, image_state2.real_image)


def _psnr(image1: PIL.Image.Image, image2: PIL.Image.Image) -> float:
    for image in (image1, image2):
        if image.mode not in {'L', 'RGB'}:
            raise ValueError(
                'Cannot calculate PSNR for {image.mode} image mode. '
                'Supported image modes are: {PSNR_SUPPORTED_MODES}.'
            )

    if image1.mode != image2.mode:
        raise ValueError(
            'Cannot calculate PSNR for images of different modes. '
            f'Left image has mode {image1.mode}, but right image has mode {image2.mode}.'
        )
    if image1.size != image2.size:
        raise ValueError(
            'Cannot calculate PSNR for images of different sizes. '
            f'Left image has size {image1.size}, but right image has size {image2.size}.'
        )

    a1 = np.asarray(image1)
    a2 = np.asarray(image2)

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


def to_grayscale(image_state: ImageState, method: str) -> ImageState:
    image = _to_grayscale(image_state.real_image, method=method)
    return ImageState(visible_image=image, real_image=image)


TO_GRAYSCALE_METHODS = ['mean', 'CCIR 601-1']


def _to_grayscale(image: PIL.Image.Image, method: str) -> PIL.Image.Image:
    if image.mode == 'L':
        return image

    if image.mode != 'RGB':
        raise ValueError(
            'Cannot convert image in mode {image.mode} to grayscale. '
            'Image mode must be RGB.'
        )

    if method not in TO_GRAYSCALE_METHODS:
        raise ValueError(
            f'Unknown method {method} for conversion to grayscale.'
    )

    if method == 'mean':
        return _to_grayscale_mean(image)
    return _to_grayscale_ccir601(image)


def _to_grayscale_mean(image: PIL.Image.Image) -> PIL.Image.Image:
    array = np.asarray(image)
    grayscale_array = (array.sum(axis=2) / 3).astype(np.uint8)
    grayscale_image = PIL.Image.fromarray(grayscale_array, mode='L')
    grayscale_image.filename = image.filename
    return grayscale_image


def _to_grayscale_ccir601(image: PIL.Image.Image) -> PIL.Image.Image:
    array = np.asarray(image).astype(int)
    grayscale_array = ((77 * array[:,:,0] + 150 * array[:,:,1] + 29 * array[:,:,2]) / 256).astype(np.uint8)
    grayscale_image = PIL.Image.fromarray(grayscale_array, mode='L')
    grayscale_image.filename = image.filename
    return grayscale_image


