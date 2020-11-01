import io
import os.path

import PySimpleGUI as sg

import core
from core import ImageValueError


BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DEFAULT_IMAGE1_FILEPATH = os.path.join(BASE_DIR, 'resources/test_images/image_Peppers512rgb.png')
IMAGE_SIZE = (512, 512)


def main():
    # create layout
    image1 = image2 = core.upload_image(DEFAULT_IMAGE1_FILEPATH)
    layout = setup_layout(image1, image2)
    
    # setup window
    sg.theme('Default1')
    window = sg.Window('Image C&C', layout, finalize=True)
    window.move(0, 0)
    
    # handle events
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        # chose working image
        if values['working_image_dropdown'] == 'left':
            working_image = image1
            working_image_element_key = 'image1_element'
        else:
            working_image = image2
            working_image_element_key = 'image2_element'

        # image1 events
        if event == 'upload_image1_filepath':
            try:
                image1 = core.upload_image(values['upload_image1_filepath'])
            except ValueError as e:
                sg.popup(e.args[0])
                continue

            update_image_element(window, 'image1_element', image1)

            # update working image
            if values['working_image_dropdown'] == 'left':
                working_image = image1
        elif event == 'save_as_image1_dummy_input':
            filepath = values['save_as_image1_dummy_input']
            if filepath == '':
                continue
            try:
                core.save_image(image1, filepath)
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

        # image2 events
        elif event == 'upload_image2_filepath':
            try:
                image2 = core.upload_image(values['upload_image2_filepath'])
            except ValueError as e:
                sg.popup(e.args[0])
                continue
            update_image_element(window, 'image2_element', image2)

            # update working image
            if values['working_image_dropdown'] == 'right':
                working_image = image2
        elif event == 'save_as_image2_dummy_input':
            filepath = values['save_as_image2_dummy_input']
            if filepath == '':
                continue
            try:
                core.save_image(image2, filepath)
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

        # comparison events
        elif event == 'calculate_psnr_button':
            try:
                psnr = core.psnr(image1, image2)
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

            if psnr == float('inf'):
                result = '∞'
            else:
                result = f'{psnr:.4f}'
            window['calculate_psnr_result'].update(result)

        # instruments events
        elif event == 'convert_to_grayscale_button':
            method = values['convert_to_grayscale_method_dropdown']
            try:
                working_image = core.to_grayscale(working_image, method=method)
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

            update_image_element(window, working_image_element_key, working_image)
        elif event == 'rgb_to_ycbcr_button':
            try:
                working_image = core.rgb_to_ycbcr(working_image)
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

            update_image_element(window, working_image_element_key, working_image)
        elif event == 'y_button':
            try:
                working_image = core.ycbcr_channel_as_grayscale_image(working_image, channel='Y')
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

            update_image_element(window, working_image_element_key, working_image)
        elif event == 'cb_button':
            try:
                working_image = core.ycbcr_channel_as_grayscale_image(working_image, channel='Cb')
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

            update_image_element(window, working_image_element_key, working_image)
        elif event == 'cr_button':
            try:
                working_image = core.ycbcr_channel_as_grayscale_image(working_image, channel='Cr')
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

            update_image_element(window, working_image_element_key, working_image)
        elif event == 'ycbcr_to_rgb_button':
            try:
                working_image = core.ycbcr_to_rgb(working_image)
            except ImageValueError as e:
                sg.popup(e.args[0])
                continue

            update_image_element(window, working_image_element_key, working_image)

        # if working image has changed, update left or right image accordingly
        if values['working_image_dropdown'] == 'left':
            image1 = working_image
        else:
            image2 = working_image


def setup_layout(default_image1, default_image2):
    image1_frame = setup_image_frame('image1', default_image1)
    image2_frame = setup_image_frame('image2', default_image2)
    instruments_frame = setup_instruments_frame()
    comparison_frame = setup_comparison_frame()

    return [
        [image1_frame, image2_frame, instruments_frame],
        [comparison_frame]
    ]


def setup_image_frame(image_name, default_image):
    image_element = sg.Image(
        data=core.image_to_bytes(default_image),
        key=f'{image_name}_element',
        size=IMAGE_SIZE
    )
    upload_image_button = sg.FileBrowse(
        'Upload image', target=f'upload_{image_name}_filepath'
    )
    upload_image_filepath = sg.Input(
        core.image_filename(default_image),
        key=f'upload_{image_name}_filepath',
        disabled=True, enable_events=True,
    )

    # this is invisible input to save filename to when user saves an image
    save_as_image_dummy_input = sg.Input(
        '', key=f'save_as_{image_name}_dummy_input',
        visible=False, enable_events=True,
    )
    save_as_image_button = sg.SaveAs(
        'Save as', key=f'save_as_{image_name}_button',
        enable_events=True
    )
    image_frame_layout = [
        [image_element],
        [upload_image_filepath, upload_image_button, save_as_image_dummy_input, save_as_image_button]
    ]
    image_frame = sg.Frame('', image_frame_layout)
    return image_frame


def setup_instruments_frame():
    working_image_text = sg.Text('Image: ')
    working_image_dropdown = sg.Combo(
        ['left', 'right'], key='working_image_dropdown',
        default_value='left', readonly=True, enable_events=True
        , pad=(0, 10)
    )

    convert_to_grayscale_button = sg.Button(
        'Convert to grayscale', key='convert_to_grayscale_button'
    )
    convert_to_grayscale_text = sg.Text('Method: ')
    convert_to_grayscale_method_dropdown = sg.Combo(
        core.TO_GRAYSCALE_METHODS, key='convert_to_grayscale_method_dropdown',
        default_value=core.TO_GRAYSCALE_METHODS[0], readonly=True,
    )
    convert_to_grayscale_column = sg.Column([
        [convert_to_grayscale_button],
        [convert_to_grayscale_text, convert_to_grayscale_method_dropdown],
    ], pad=(0, 10))

    rgb_to_ycbcr_button = sg.Button(
        'Convert RGB to YCbCr', key='rgb_to_ycbcr_button'
    )
    y_button = sg.Button('Y', key='y_button')
    cb_button = sg.Button('Cb', key='cb_button')
    cr_button = sg.Button('Cr', key='cr_button')
    ycbcr_to_rgb_button = sg.Button(
        'Convert YCbCr to RGB', key='ycbcr_to_rgb_button'
    )
    ycbcr_column = sg.Column([
        [rgb_to_ycbcr_button],
        [y_button, cb_button, cr_button],
        [ycbcr_to_rgb_button]
    ], pad=(0, 10))

    instruments_frame_layout = [
        [working_image_text, working_image_dropdown],
        [convert_to_grayscale_column],
        [ycbcr_column]
    ]
    instruments_frame = sg.Frame('', instruments_frame_layout)

    return instruments_frame


def setup_comparison_frame():
    calculate_psnr_button = sg.Button(
        'Calculate PSNR', key='calculate_psnr_button'
    )
    calculate_psnr_result = sg.Text('', key='calculate_psnr_result', size=(10, 1))
    comparison_frame_layout = [
        [calculate_psnr_button],
        [calculate_psnr_result],
    ]

    comparison_frame = sg.Frame('', comparison_frame_layout)

    return comparison_frame


def update_image_element(window, image_element_key, image):
    window[image_element_key].update(
        data=core.image_to_bytes(image), size=IMAGE_SIZE
    )