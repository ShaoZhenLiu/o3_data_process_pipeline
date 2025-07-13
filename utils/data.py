import json
import random
from collections import defaultdict

def exploring_data(data_ls):
    """
    Explore the data to find the number of images and the number of questions.
    """
    num_images = len(data_ls)
    
    # 遍历前五条数据的指定字段
    for data in data_ls[:5]:
        index = data["id"]
        image_path = data["image_path"]
        decision = data["decision"]
        print(f"Index: {index}, Image Path: {image_path}, Decision: {decision}")
        
        
def load_dataset(data_path, range_limit=None):
    """
    Load the dataset from the specified JSON file.
    """
    from datasets import load_dataset
    
    dataset = load_dataset('json', data_files=data_path, split='train')
    if range_limit is not None:
        if isinstance(range_limit, int):
            dataset = dataset.select([i for i in list(range(range_limit)) if i < len(dataset)])
        else:
            dataset = dataset.select(range(range_limit[0], range_limit[1]))
        
    return dataset


def fliter_dataset(dataset, limit):
    """
    统计所有operation种类及数量，并从每个类别中各选10个数据组成新的数据集，总数不超过limit。
    """

    # 统计operation种类及数量
    operation_count = defaultdict(int)
    operation_indices = defaultdict(list)

    for idx, data in enumerate(dataset):
        op = data["decision"]["operation"]
        operation_count[op] += 1
        operation_indices[op].append(idx)

    print("Operation种类及数量：")
    for op, count in operation_count.items():
        print(f"{op}: {count}")

    # 从每个类别中各选20个数据
    selected_indices = []
    for op, indices in operation_indices.items():
        selected_indices.extend(indices[:20])

    # 如果总数超过limit，则随机采样limit个
    if len(selected_indices) > limit:
        random.seed(42)
        selected_indices = random.sample(selected_indices, limit)

    # 返回新的数据集
    new_dataset = dataset.select(selected_indices)
    return new_dataset


def save_dataset(dataset, output_code_data_path):
    # 将dataset保存为一个json列表
    data_list = [item for item in dataset]
    with open(output_code_data_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4, default=str)
        print(f"dataset save to {output_code_data_path}")