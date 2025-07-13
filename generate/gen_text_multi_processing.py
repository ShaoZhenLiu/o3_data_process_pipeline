import os
import json
import requests
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from tqdm import tqdm
from openai import OpenAI

from utils.image_utils import encode_image_to_base64
from utils.prompt import think_prompt, mistake_correct_instruction

url = "http://0.0.0.0:18901/v1"
key = "EMPTY"
client = OpenAI(
    base_url=url,
    api_key=key,
)
try:
    response = requests.get(f"{url}/models")
    eval_model_name = response.json()['data'][0]['id']
except Exception as e:
    print(e)


def create_message(user_query, base64_image, **kwargs):
    message = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can generate code based on images and text descriptions.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}"} },
                {"type": "text", "text": user_query},
            ],
        }
    ]
    if kwargs["turn"] > 0 and kwargs["error"] is not None:
        message.extend([
            {
                "role": "assistant",
                "content": kwargs["code"],
            },
            {
                "role": "user",
                "content": mistake_correct_instruction.replace("{{ error_message }}", kwargs["error"])
            },
        ])
    return message


def process(args):
    user_query, image_path, turn_number, error_message, code = args
    base64_image = encode_image_to_base64(image_path=image_path)
    message = create_message(user_query, base64_image, turn=turn_number, error=error_message, code=code)
    
    if "keye" in eval_model_name.lower():
        if isinstance(message[-1]["content"], str):
            message[-1]["content"] += think_prompt
        elif isinstance(message[-1]["content"], list):
            message[-1]["content"][-1]["text"] += think_prompt
        
    try:
        params = {
            "model": eval_model_name,
            "messages": message,
            "max_tokens": 10240,
            "temperature": 0.0,
            "top_p": 0.95,
        }
        response = client.chat.completions.create(**params)
        output_text = response.choices[0].message.content
        return output_text
    except Exception as e:
        print(f"Error processing text: {e}")
        return None


def vllm_client_generate(user_query_ls, image_paths, num_workers=8, **kwargs):
    turn_number = kwargs["turn_number"]
    error_message_ls = kwargs["error_message_ls"]
    processed_image_path_ls = kwargs["processed_image_path_ls"]
    codes = kwargs["codes"]
    
    process_args = [
        [
            user_query_ls[i], 
            image_paths[i],
            turn_number,
            error_message_ls[i] if turn_number > 0 else None,
            codes[i] if turn_number > 0 else None,
        ] for i in range(len(user_query_ls))
    ]
    
    # 把这边的第二轮之后对再实现一下
    pool = multiprocessing.Pool(processes=num_workers)
    save_json = []
    with tqdm(total=len(user_query_ls), desc="Processing text for each image") as pbar:
        for result in pool.imap(process, process_args):
            if result is not None:
                save_json.append(result)
                pbar.update(1)

    pool.close()
    pool.join()
    return save_json
