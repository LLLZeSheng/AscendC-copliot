import os
import re
import hashlib
import logging


logger = logging.getLogger("eval_operator")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]

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
        logger.error("REPLACE_ERR: Target file not found: %s", target_file_path)
        return None

    try:
        with open(target_file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
    except Exception as e:
        logger.error("REPLACE_ERR: Failed to read target file: %s", e)
        return None

    # 2. Get the list of replacements
    replacements = read_generated_sol(file_generated_path)
    target_marker_count = current_content.count("REPLACE_START")
    logger.info(
        "REPLACE_BEGIN: target=%s generated=%s target_len=%d target_sha=%s target_markers=%d",
        target_file_path,
        file_generated_path,
        len(current_content),
        _short_hash(current_content),
        target_marker_count,
    )
    if not replacements:
        logger.warning("REPLACE_NO_REPLACEMENTS: generated file had no marker blocks.")
        return None

    logger.info("REPLACE_FOUND: %d replacement(s) to apply.", len(replacements))

    # 3. Apply updates to the string in memory
    applied = 0
    for rep_data in replacements:
        current_content, did_apply = update_string_with_markers(current_content, rep_data)
        if did_apply:
            applied += 1

    if applied == 0:
        logger.warning("REPLACE_APPLY_NONE: No markers matched in target file.")
        return None

    logger.info("REPLACE_APPLIED: %d/%d replacements applied.", applied, len(replacements))
    logger.info(
        "REPLACE_RESULT: result_len=%d result_sha=%s",
        len(current_content),
        _short_hash(current_content),
    )
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
        return content, False

    # Define the markers
    start_marker = f"// [[[ REPLACE_START: {func_name} ]]]"
    end_marker = "// [[[ REPLACE_END ]]]"

    # 1. Find START marker
    start_index = content.find(start_marker)
    if start_index == -1:
        logger.warning("REPLACE_SKIP: Marker not found: %s", start_marker)
        return content, False

    # 2. Find END marker (searching strictly after the start marker)
    end_index = content.find(end_marker, start_index)
    if end_index == -1:
        logger.error("REPLACE_ERR: Missing end marker for function '%s'", func_name)
        return content, False

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
    
    logger.info("REPLACE_OK: Updated code for %s (new_len=%d).", func_name, len(new_code))
    return updated_content, True
