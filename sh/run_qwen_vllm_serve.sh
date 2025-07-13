set -x

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 自动获取GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

MODEL_PATH="/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-VL-72B-Instruct"
# MODEL_PATH_KEYE="/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Keye-VL-8B-Preview"
MODEL_NAME="qwen2.5-vl-72b-instruct"


vllm serve $MODEL_PATH \
    --port 18901 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size $NUM_GPUS \
    --dtype bfloat16 \
    --limit-mm-per-prompt "image=10" \
    --disable-log-requests \
    --served-model-name $MODEL_NAME \
    --trust-remote-code
