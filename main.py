import os
import glob
import json
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import requests

from utils.model_utils import load_model
from utils.data import load_dataset, fliter_dataset, save_dataset
from generate.gen_text import generate_text
from generate.text2code import text_to_code
from generate.sandbox_result import running_code, running_code_multi_processing


# model_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Keye-VL-8B-Preview"
# model_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-VL-72B-Instruct"
data_path = "/apdcephfs_gy5/share_303588738/yingzhepeng/results/O3/need_processing_image_qa/needs_processing_data.json"
output_img_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/processed_image"
# output_img_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/processed_image_subset100_keye"
# output_img_path = "./img"
output_code_data_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data.json"
# output_code_data_path = "./code_processed_data.json"
# output_code_data_path = "./code_processed_data_100_test.json"

if __name__ == "__main__":
    use_vllm = True  # Set to True if you want to use vLLM for inference
    use_server = True
    
    dataset = load_dataset(data_path, range_limit=None)
    print(len(dataset))

    model, processor = load_model(model_path, use_vllm=use_vllm) if use_server == False else (None, None)

    # 查看处理不同任务的能力
    # dataset = fliter_dataset(dataset, limit=100)
    
    for i in range(2):
        print(f"Turn {i} start.")
        print("start to generate text for code")
        dataset = dataset.map(
            generate_text,
            fn_kwargs={
                'model': model,
                'processor': processor,
                'use_vllm': use_vllm,
                'use_server': use_server,
                'turn_number': i,
            },
            batched=True,
            batch_size=4 if use_server == False is None else len(dataset),
            desc="Generating text for each image",
            load_from_cache_file=False,  # 如果不设置，可能会使用之前的结果进而跳过map
        )
        
        print("extracting code")
        dataset = dataset.map(
            lambda x: {"code": text_to_code(x[f"code_infer_text_turn{i}"])},
            batched=False,
            # remove_columns=["code_infer_text"],
            desc="Converting generated text to code",
            load_from_cache_file=False,
        )
        
        save_dataset(dataset, output_code_data_path)
        
        print("execute code")  # 这一步容易出错
        # dataset = dataset.map(
        #     lambda x: {
        #         "image_path_processed": running_code(
        #             x["code"], 
        #             kernel="python3", 
        #             output_path=os.path.join(
        #                 output_img_path, 
        #                 f"{Path(x["image_path"]).stem}_output.jpg",
        #             )
        #         ) if x["code"] else None},
        #     batched=False,
        #     desc="running code",
        #     load_from_cache_file=False,
        # )
        dataset = dataset.map(
            running_code_multi_processing,
            fn_kwargs={
                'output_img_path': output_img_path,
            },
            batched=True,
            batch_size=len(dataset),
            desc="running code",
            load_from_cache_file=False,
        )
        
        save_dataset(dataset, output_code_data_path)
        
        # 过滤一下，只留下代码运行错误的数据，进入下一轮
        dataset = dataset.filter(lambda x: x["code_error_message"] not in [None, ""])
        # 验证过滤后的dataset中code_error_message字段均不为空
        print(f"After turn {i}, we have: {len(dataset)}")
        
        debug = True
        if debug:
            output_code_data_path_filter = output_code_data_path.replace(".json", "_filter.json")
            save_dataset(dataset, output_code_data_path_filter)
            if len(dataset) != 0:
                assert all(x.get("code_error_message") not in [None, ""] for x in dataset), "Some code_error_message fields are empty!"
        output_code_data_path = output_code_data_path.replace(".json", f"_{i+1}.json")
    
    # 初始版本，记得改改
    with open("/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data.json", "r") as f:
        data = json.load(f)
    with open("/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data_1.json", "r") as f:
        data_1 = json.load(f)

    data_dict = {item["id"]: item for item in data}
    for item in data_1:
        data_dict[item["id"]] = item

    merged_data = list(data_dict.values())

    with open("/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data_merged.json", "w") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data.json"
    "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data_1.json"
    