import io

import PIL.Image


def upload_image(filepath):
    image = PIL.Image.open(filepath)
    return image


def image_to_bytes(image):
    bio = io.BytesIO()
    image.save(bio, format='PNG')
    return bio.getvalue()


def save_image(filepath, image):
    image.save(filepath, 'PNG')