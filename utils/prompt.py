task_instruction_old = """Operation: {{ operation }}
Condition: {{ condition }}
Reason: {{ reason }}
Generate image executing code based on the given image, and description text using python.
* image_path = "{{ image_path }}", image_size = {{ image_size }}.
* Since we will execute the code in a Jupyter environment, you must to use `image.show()` to display the image at the end fo the code.
* **DON'T SAVE THE IMAGE TO A  FILE**, just display it.
* If you are using opencv to process image, you must turn the processed image into PIL format and show it, instead of show it using opencv directly.

Example:
<write some examples here>
"""

task_instruction = """Generate Python code to process an image according to the specifications below. Your code MUST:
1. Be executed in a Jupyter environment
2. Display the result using `image.show()` at the end
3. NOT save any file to disk
4. Convert OpenCV images to PIL format before display

Required Information:
- Operation: {{ operation }}
- Condition: {{ condition }}
- Reason: {{ reason }}
- Image Path: "{{ image_path }}"
- Image Size: {{ image_size }}

Code Requirements:
a) Begin by loading the image
b) Implement the specified operation with condition-based logic
c) If using OpenCV:
   - Convert BGR→RGB before PIL conversion
   - Use `Image.fromarray(cv2_image_rgb)`
d) Final line must be: `processed_image.show()`"""
# e) Don't use package named pytesseract"""

mistake_correct_instruction = """Your code has an error, here is the error message, please fix it:
{{ error_message }}"""

no_code_prompt = "Your previous response doesn't contain any code. Please answer it again and generate some code."

missing_error_prompt = "An error appeared but no exception raise, it seems like you are using `try ... except`"

think_prompt = "/think"

code_marker = "```python"