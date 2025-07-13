import os
import re
import json
import requests
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from tqdm import tqdm

from utils.image_utils import decode_base64_to_image
from utils.prompt import no_code_prompt, missing_error_prompt


counter = 0
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def combine_cells(code, method=None):
    if method == "small":
        return code.split("\n")
    if method == "middle":
        return code.split("\n\n")
    return [code]


def request_jupyter(cells, kernel='python3'):
    """
    Send a request to the Jupyter server to run the provided cells.
    """
    response = requests.post('http://localhost:8080/run_jupyter', json={
        'cells': cells,
        'kernel': kernel
    })
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    

def print_error_message(response_json):
    print("No image output found in any cell.")
    print("Response:", response_json["driver"]["stdout"])
    error_ls = []
    for cell in response_json["cells"]:
        if len(cell["error"]) == 0:
            continue
        # error_message_full = "\n".join(cell["error"][0]["traceback"])
        error_message_full = ansi_escape.sub('', "\n".join(cell["error"][0]["traceback"]))
        print(error_message_full)
        error_ls.append(error_message_full)
        # print(cell["error"][0]["ename"])
        # print(cell["error"][0]["evalue"])
        # error_ls.append(f"{cell['error'][0]['ename']}: {cell['error'][0]['evalue']}")
    print("-" * 20)
    return "\n\n".join(error_ls) if len(error_ls) > 0 else missing_error_prompt


def running_code(code, kernel='python3', output_path=None):
    """
    Run the provided code in a Jupyter kernel and save the output image if specified.
    """
    global counter
    cells = combine_cells(code)
    response_json = request_jupyter(cells, kernel=kernel)

    # Extract the last cell's display output
    # 从最后一个 cell 向前遍历，找到第一个包含 image/png 的 cell
    found_image = False
    if response_json['cells']:
        for cell in reversed(response_json['cells']):
            display_output = cell.get('display', [])
            if display_output and isinstance(display_output, list):
                for output in display_output:
                    if 'image/png' in output:
                        returned_image = output['image/png']
                        decode_base64_to_image(returned_image, output_path)
                        found_image = True
                        break
            if found_image:
                break
        if not found_image:
            print_error_message(response_json)
            counter += 1
            print(f"current error count: {counter}")
    else:
        print("No cells found in the response.")
        print("Response JSON:", json.dumps(response_json, indent=2))
        counter += 1
        print(f"current error count: {counter}")
    
    return output_path


def process(args):
    code, kernel, output_path = args
    if code is None:
        print("No code found in the previous request.")
        return None, no_code_prompt
    
    cells = combine_cells(code)
    response_json = request_jupyter(cells, kernel=kernel)

    # Extract the last cell's display output
    # 从最后一个 cell 向前遍历，找到第一个包含 image/png 的 cell
    found_image = False
    if response_json['cells']:
        for cell in reversed(response_json['cells']):
            display_output = cell.get('display', [])
            if display_output and isinstance(display_output, list):
                for output in display_output:
                    if 'image/png' in output:
                        returned_image = output['image/png']
                        decode_base64_to_image(returned_image, output_path)
                        found_image = True
                        break
            if found_image:
                break
        if not found_image:
            error_meassge = print_error_message(response_json)
            return None, error_meassge
    else:
        print("No cells found in the response.")
        print("Response JSON:", json.dumps(response_json, indent=2))
    
    return output_path, None


def _running_code_multi_processing(codes, kernel='python3', output_path_ls=None, num_workers=8):
    process_args = [
        [codes[i], kernel, output_path_ls[i]] for i in range(len(codes))
    ]
    pool = multiprocessing.Pool(processes=num_workers)
    img_output_ls = []
    error_msg_ls = []
    with tqdm(total=len(codes), desc="Processing text for each image") as pbar:
        for result in pool.imap(process, process_args):
            if result is not None:
                img_output_ls.append(result[0])
                error_msg_ls.append(result[1])
                pbar.update(1)

    pool.close()
    pool.join()
    return img_output_ls, error_msg_ls

def running_code_multi_processing(examples, output_img_path):
    codes = examples["code"]
    output_path_ls = [os.path.join(output_img_path, f"{Path(path).stem}_output.jpg") for path in examples["image_path"]]
    
    img_output_path_ls, error_message_ls = _running_code_multi_processing(codes, kernel="python3", output_path_ls=output_path_ls)
    
    examples["image_path_processed"] = img_output_path_ls
    # print(img_output_path_ls)
    examples["code_error_message"] = error_message_ls
    # print(error_message_ls)
    return examples


if __name__ == "__main__":
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

    response_json = request_jupyter(cells, kernel='python3')

    # print(response.text)

    returned_image = response_json['cells'][-1]['display'][-1]['image/png']

    # 将图片从base64解码
    image_data = base64.b64decode(returned_image)

    # 将解码后的数据保存为图片文件
    with open('./img/imgoutput_image.png', 'wb') as f:
        f.write(image_data)
    print("Image saved as './img/output_image.png'")