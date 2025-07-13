import jinja2
from keye_vl_utils import process_vision_info as keye_vl_process_vision_info
from qwen_vl_utils import process_vision_info as qwen_vl_process_vision_info
from vllm import SamplingParams

from utils.prompt import task_instruction, think_prompt
from utils.image_utils import get_image_size, encode_image_to_base64
from generate.gen_text_multi_processing import vllm_client_generate 


def generate_text(
    examples, 
    model=None, 
    processor=None, 
    use_vllm=False, 
    use_server=False, 
    turn_number=0,
    ):
    """
    Generate text based on the data.
    """
    template = jinja2.Template(task_instruction)
    
    indexes = examples["id"]
    image_paths = examples["image_path"]
    decisions = [decision for decision in examples["decision"]]
    operations = [decision["operation"] for decision in decisions]
    conditions = [decision["parameters"] for decision in decisions]
    reasons = [decision["reason"] for decision in decisions]
    image_size = [tuple(img_data["resolution"]) for img_data in examples["image"]]
    
    # 之后多轮需要的数据
    error_message_ls = examples["code_error_message"] if turn_number > 0 else None
    processed_image_path_ls = examples["image_path_processed"] if turn_number > 0 else None
    codes = examples["code"] if turn_number > 0 else None
    
    user_query_ls = [
        template.render(
            operation=operations[i], 
            condition=conditions[i], 
            reason=reasons[i],
            image_path=image_paths[i],
            # image_size=get_image_size(image_paths[i]),
            image_size=image_size[i],
        )
        for i in range(len(indexes))
    ]
    # print(user_query_ls[0])
    
    if use_server:  # 有服务器，使用多线程对服务器请求
    #     base64_images = [
    #         encode_image_to_base64(image_path) for image_path in image_paths
    #     ]
    #     messages = [
    #         [
    #             {
    #                 "role": "system",
    #                 "content": "You are a helpful assistant that can generate code based on images and text descriptions.",
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_images[i]}"} },
    #                     {"type": "text", "text": user_query_ls[i] + think_prompt},
    #                 ],
    #             }
    #         ] for i in range(len(indexes))
    #     ]
        output_texts = vllm_client_generate(
            user_query_ls, 
            image_paths, 
            num_workers=8,
            error_message_ls=error_message_ls,
            processed_image_path_ls=processed_image_path_ls,
            codes=codes,
            turn_number=turn_number,
        )
    else:  # 没有服务器，使用本地模型
        messages = [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can generate code based on images and text descriptions.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_paths[i]},
                        {"type": "text", "text": user_query_ls[i] + think_prompt},
                    ],
                }
            ] for i in range(len(indexes))
        ]

        # Preparation for batch inference
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        if use_vllm:
            output_texts = vllm_generate(model, texts, messages)
        else:
            output_texts = transformers_generate(model, processor, texts, messages)

    # print(type(output_texts), len(output_texts))  # <class 'list'> 4
    # print(output_texts[0], type(output_texts[0]))  # <class 'str'>

    examples[f"code_infer_text_turn{turn_number}"] = output_texts

    return examples


def vllm_generate(llm, texts, messages):
    # For vLLM, we need to process vision info differently
    image_inputs, video_inputs = qwen_vl_process_vision_info(messages)
    mm_data = [
        {
            "image": image_inputs[i] if image_inputs is not None else None,
            # "video": video_inputs[i] if video_inputs is not None else None,
        } for i in range(len(texts))
    ]
    llm_inputs_ls = [
        {
            "prompt": texts[i],
            "multi_modal_data": mm_data[i],
        } for i in range(len(texts))
    ]
    outputs = llm.generate(
        llm_inputs_ls,
        sampling_params=SamplingParams(
            max_tokens=10240,
            temperature=0.1,
            top_p=0.95,
            # top_k=40,
        ),
        use_tqdm=False,
    )
    output_texts = [
        output.outputs[0].text for output in outputs
    ]
    return output_texts


def transformers_generate(model, processor, texts, messages):
    # For transformers, we use the Keye-VL processing function
    image_inputs, video_inputs = keye_vl_process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=10240)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_texts