import json
import base64
import requests

image_path = '/mnt/private_berlinni/liushaozhen/Img2Code/img/image1.png'
crop_box = '(0, 200, 400, 600)'

cells = [
    '''
from PIL import Image

# Load the image
image_path = "/mnt/private_berlinni/liushaozhen/Img2Code/img/image0.png"
img = Image.open(image_path)

# Define the cropping coordinates (x1, y1, x2, y2)
# Focus on the faces by cropping the upper part of the image
crop_box = (0, 0, 2628, 600)  # Adjust the y2 value as needed to focus on the faces

# Crop the image
cropped_img = img.crop(crop_box)

# Display the cropped image
cropped_img.show()
    ''',
]

response = requests.post('http://localhost:8080/run_jupyter', json={
    'cells': cells,
    'kernel': 'python3'
})

# print(response.text)

returned_image = response.json()['cells'][-1]['display'][-1]['image/png']

# 将图片从base64解码
image_data = base64.b64decode(returned_image)

# 将解码后的数据保存为图片文件
with open('./img/imgoutput_image.png', 'wb') as f:
    f.write(image_data)
print("Image saved as './img/output_image.png'")