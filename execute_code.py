import os
import json
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import requests

from utils.data import load_dataset
from generate.sandbox_result import running_code, running_code_multi_processing


# model_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Keye-VL-8B-Preview"
# model_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-VL-72B-Instruct"
# data_path = "/apdcephfs_gy5/share_303588738/yingzhepeng/results/O3/need_processing_image_qa/needs_processing_data.json"
output_img_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/processed_image_subset100_keye"
# output_img_path = "./img"
output_code_data_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data_subset100_keye.json"
# output_code_data_path = "./code_processed_data_100_test.json"

if __name__ == "__main__":
    use_vllm = True  # Set to True if you want to use vLLM for inference
    use_server = True
    
    # 加载之前处理好的数据集
    dataset = load_dataset(output_code_data_path, range_limit=None)

    print("execute code")
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
    # )
    dataset = dataset.map(
        running_code_multi_processing,
        fn_kwargs={
            'output_img_path': output_img_path,
        },
        batched=True,
        batch_size=len(dataset),
        desc="running code",
        load_from_cache_file=False,  # 如果不设置，可能会使用之前的结果进而跳过map
    )
    # 将dataset保存为一个json列表
    # data_list = [item for item in dataset]
    # with open(output_code_data_path, "w", encoding="utf-8") as f:
    #     json.dump(data_list, f, ensure_ascii=False, indent=4, default=str)
    #     print(f"dataset save to {output_code_data_path}")
    
    
    