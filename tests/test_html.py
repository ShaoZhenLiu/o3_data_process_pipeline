from datasets import load_dataset
from PIL import Image
import base64
import io
import os

from tqdm import tqdm

def image_to_base64(image_path):
    """Convert image to base64 encoded string"""
    if not os.path.exists(image_path):
        return None
    
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_html_for_item(item):
    """Generate HTML for a single dataset item"""
    # Handle image display
    image_html = ""
    if 'image_path' in item and item['image_path']:
        image_base64 = image_to_base64(item['image_path'])
        if image_base64:
            image_html = f'<div class="image-container"><h3>Original Image</h3><img src="data:image/jpeg;base64,{image_base64}" alt="Original image"></div>'
    
    if 'image_path_processed' in item and item['image_path_processed']:
        processed_base64 = image_to_base64(item['image_path_processed'])
        if processed_base64:
            image_html += f'<div class="image-container"><h3>Processed Image</h3><img src="data:image/jpeg;base64,{processed_base64}" alt="Processed image"></div>'
    
    # Generate HTML for each field
    html_parts = []
    for key, value in item.items():
        if key in ['image_path', 'image_path_processed']:
            continue  # Already handled above
            
        if isinstance(value, dict):
            value_str = "<ul>" + "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in value.items()) + "</ul>"
        else:
            value_str = str(value)
            
        html_parts.append(f"""
        <div class="field">
            <h3>{key}</h3>
            <div class="value">{value_str}</div>
        </div>
        """)
    
    return f"""
    <div class="dataset-item">
        <h2>Data Item: {item.get('id', 'N/A')}</h2>
        {image_html}
        {"".join(html_parts)}
        <hr>
    </div>
    """

def generate_full_html(dataset):
    """Generate complete HTML for the dataset"""
    item_html= []
    err_count = 0
    for item in tqdm(dataset):
        try:
            data_text = generate_html_for_item(item)
            item_html.append(data_text)
        except Exception as e:
            print(e)
            err_count += 1
    
    print(err_count)
    items_html = "\n".join(item_html)
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dataset Viewer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .dataset-item {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .field {{
                margin-bottom: 15px;
            }}
            .field h3 {{
                margin-bottom: 5px;
                color: #2c3e50;
            }}
            .value {{
                background-color: white;
                padding: 10px;
                border-radius: 3px;
                border-left: 4px solid #3498db;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .image-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                max-height: 500px;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }}
            code {{
                background-color: #f0f0f0;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }}
            pre {{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 3px;
                overflow-x: auto;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Dataset Viewer</h1>
        <div class="dataset-container">
            {items_html}
        </div>
    </body>
    </html>
    """

def save_dataset_as_html(dataset, output_path="dataset_viewer.html"):
    """Save dataset as a standalone HTML file"""
    html_content = generate_full_html(dataset)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML file saved to {output_path}")

# Example usage:
# Assuming you have loaded your dataset
# dataset = load_dataset(...)  # Your dataset loading code here
# save_dataset_as_html(dataset)

data_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/O3/code_processed_data_subset100_keye.json"
dataset = load_dataset('json', data_files=data_path, split='train')

save_dataset_as_html(dataset, "./keye_code_sample_100.html")