import base64
from PIL import Image


def get_image_size(image_path):
    """
    Get the size of the image.
    """    
    img = Image.open(image_path)
    return img.size


def decode_base64_to_image(base64_str, output_path=None):
    """
    Convert a base64 string to an image and save it to the specified path.
    """
    image_data = base64.b64decode(base64_str)
    if output_path is not None:
        with open(output_path, 'wb') as f:
            f.write(image_data)
        # print(f"Image saved as '{output_path}'")
    return image_data


def encode_image_to_base64(image_path):
    """
    Convert an image to a base64 string.
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')