import os
import re

def replace(target_file_path, file_generated_path):
    """
    Main driver:
    1. Reads target file content.
    2. Parses generated solutions.
    3. Iteratively updates the content string.
    4. Returns the final string.
    """
    # 1. Read the Target File strictly once
    if not os.path.exists(target_file_path):
        print(f"Error: Target file '{target_file_path}' not found.")
        return None

    try:
        with open(target_file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
    except Exception as e:
        print(f"Error reading target file: {e}")
        return None

    # 2. Get the list of replacements
    replacements = read_generated_sol(file_generated_path)
    
    print(current_content)
    if not replacements:
        print("No replacements found or error reading generated file.")
        return current_content

    print(f"Found {len(replacements)} replacement(s). Applying...")

    # 3. Apply updates to the string in memory
    for rep_data in replacements:
        current_content = update_string_with_markers(current_content, rep_data)
    print(current_content)
    # 4. Return the final modified string
    return current_content


def read_generated_sol(file_generated_path):
    """
    Parses the generated file format to extract function names and code bodies.
    Returns: list of dicts [{"name": str, "code": str}]
    """
    if not os.path.exists(file_generated_path):
        print(f"Error: Generated file '{file_generated_path}' not found.")
        return []

    try:
        with open(file_generated_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading generated file: {e}")
        return []

    solutions = []

    # Regex explanation:
    # 1. Capture function name inside REPLACE_START
    # 2. Capture everything (DOTALL) until REPLACE_END
    pattern = re.compile(
        r"// \[\[\[ REPLACE_START: (.*?) \]\]\]\n(.*?)\n// \[\[\[ REPLACE_END \]\]\]", 
        re.DOTALL
    )

    matches = pattern.findall(content)

    for func_name, code_body in matches:
        solutions.append({
            "name": func_name.strip(),
            "code": code_body.strip() # Remove leading/trailing newlines from the extraction
        })

    return solutions


def update_string_with_markers(content, replacement_data):
    """
    Replaces code between specific comment anchors in a string.
    Returns the modified string.
    """
    func_name = replacement_data.get("name")
    new_code = replacement_data.get("code")

    if not func_name or new_code is None:
        return content

    # Define the markers
    start_marker = f"// [[[ REPLACE_START: {func_name} ]]]"
    end_marker = "// [[[ REPLACE_END ]]]"

    # 1. Find START marker
    start_index = content.find(start_marker)
    if start_index == -1:
        print(f"Skipping: Marker '{start_marker}' not found in content.")
        return content

    # 2. Find END marker (searching strictly after the start marker)
    end_index = content.find(end_marker, start_index)
    if end_index == -1:
        print(f"Error: Found start for '{func_name}' but missing closing marker.")
        return content

    # 3. Calculate splice positions
    # We want to insert AFTER the start marker line
    code_start_pos = start_index + len(start_marker)

    # 4. Construct the new string
    # content_before + start_marker + \n + NEW_CODE + \n + end_marker + content_after
    updated_content = (
        content[:code_start_pos] + 
        "\n" + new_code + "\n" + 
        content[end_index:]
    )
    
    print(f"Updated code for: {func_name}")
    return updated_content