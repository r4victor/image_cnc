import io
import os.path

import PySimpleGUI as sg
import PIL.ImageTk

import core


BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DEFAULT_IMAGE1_FILEPATH = os.path.join(BASE_DIR, 'resources/test_images/image_Peppers512rgb.png')


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
        print(event, values)

        if event == sg.WIN_CLOSED:
            break

        # image1 events
        if event == 'upload_image1_filepath':
            image1 = core.upload_image(values['upload_image1_filepath'])
            window['image1_element'].update(data=core.image_to_bytes(image1), size=(512, 512))
        elif event == 'save_as_image1_dummy_input':
            filepath = values['save_as_image1_dummy_input']
            core.save_image(filepath, image1)

        # image2 events
        elif event == 'upload_image2_filepath':
            image2 = core.upload_image(values['upload_image2_filepath'])
            window['image2_element'].update(data=core.image_to_bytes(image2), size=(512, 512))
        elif event == 'save_as_image2_dummy_input':
            filepath = values['save_as_image2_dummy_input']
            core.save_image(filepath, image2)
        


def setup_layout(default_image1, default_image2):
    image1_frame = setup_image_frame('image1', default_image1)
    image2_frame = setup_image_frame('image2', default_image2)

    return [
        [image1_frame, image2_frame]
    ]


def setup_image_frame(image_name, default_image):
    image_element = sg.Image(
        data=core.image_to_bytes(default_image),
        key=f'{image_name}_element',
        size=(512, 512)
    )
    upload_image_button = sg.FileBrowse(
        'Upload image', target=f'upload_{image_name}_filepath'
    )
    upload_image_filepath = sg.Input(
        default_image.filename, key=f'upload_{image_name}_filepath',
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


if __name__ == '__main__':
    main()