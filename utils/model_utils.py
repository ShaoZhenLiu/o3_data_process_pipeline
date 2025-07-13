import torch
from transformers import AutoModel, AutoProcessor


def load_model(model_path, use_vllm=False):
    """
    Load the model from the specified path.
    """
    if use_vllm:
        from vllm import LLM
        
        llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            gpu_memory_utilization=0.9,
            tensor_parallel_size=torch.cuda.device_count(),  # Adjust based on your GPU setup
            # enable_prefix_caching=True,
            seed=42,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        
        return llm, processor
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    
    return model, processor