from utils.prompt import code_marker


def text_to_code(text):
    """
    Convert the generated text to code.
    """
    if code_marker not in text:
        return None
    code = text.split(code_marker)[-1].split("```")[0].strip()    
    return code