import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from keye_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Keye-VL-8B-Preview"

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = KeyeForConditionalGeneration.from_pretrained(
#     "Kwai-Keye/Keye-VL-8B-Preview",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(model_pat, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)

image_path = "/mnt/private_berlinni/liushaozhen/Img2Code/img/image0.png"
image_size = Image.open(image_path).size
question = "Complete the matrix"
crop_description = """\"decision\": {
    \"needs_processing\": true,
    \"operation\": \"Crop\",
    \"parameters\": \"Focus on the faces of both individuals by cropping out the lower part of their bodies and some of the surrounding environment.\",
    \"reason\": \"Cropping will help to emphasize facial features that can provide clues to their ages, such as wrinkles, hair color, and overall complexion, thereby improving the estimation of the age gap between them.\"
}"""
operation = "Crop"
condition = "Focus on the faces of both individuals by cropping out the lower part of their bodies and some of the surrounding environment."
reason = "Cropping will help to emphasize facial features that can provide clues to their ages, such as wrinkles, hair color, and overall complexion, thereby improving the estimation of the age gap between them."

# crop_description = """\"decision\": {
#     \"needs_processing\": true,
#     \"operation\": \"Crop\",
#     \"parameters\": \"Crop the incomplete row (last row) and the options A through F below the matrix.\",
#     \"reason\": \"Cropping out the irrelevant portions of the image allows us to focus on the incomplete row that needs to be filled and the available options. This simplifies the process of identifying patterns within the matrix necessary to complete it.\"
# }"""
# crop_description = """\"decision\": {
#     \"needs_processing\": true,
#     \"operation\": \"Crop\",
#     \"parameters\": \"Focus on the algae and its direct connections (mussels, crabs, and plankton) while excluding other elements like the dolphin and seal.\",
#     \"reason\": \"Cropping the image to highlight the algae and its immediate consumers will make it clearer how their disappearance would affect these organisms, thus simplifying the analysis of the impact within the food web.\"
# }"""
task_instruction = f"""Operation: { operation }.
Condition: { condition }.
Reason: { reason }.
Generate image executing code based on the given image, and description text using python.
* image_path = "{ image_path }", image_size = { image_size }.
* Since we will execute the code in a Jupyter environment, you need to use img.show() to display the cropped image, instead of saving it."""

think_token = "/think"

# Auto-Thinking Mode
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": task_instruction + think_token},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=10240)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
