#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Insert C++ function markers:
  // [[[ REPLACE_START: <funcname> ]]]
  ... function ...
  // [[[ REPLACE_END ]]]

Heuristics-based lexer + parser:
- Skips strings/comments
- Detects function definitions by finding '{' preceded by a valid parameter list ')'
- Avoids control blocks (if/for/while/switch/catch/etc.)
- Avoids constructor initializer list confusion (': a(x), b(y) {')
- Handles templates, qualified names, destructors, and operators
- Skips if markers already exist around the function
- Fixes: no duplicate markers, and no markers inside function bodies (if/for/while blocks, lambdas, etc.)
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


CPP_KEYWORDS = {
    # common C++ keywords (not exhaustive, but enough for our name extraction + control-flow filtering)
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
    "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept",
    "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await", "co_return",
    "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline",
    "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator",
    "or", "or_eq", "private", "protected", "public", "register", "reinterpret_cast", "requires",
    "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
    "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid",
    "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while",
    "xor", "xor_eq",
}

CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch", "try", "do", "else"}
TYPE_BLOCK_KEYWORDS = {"class", "struct", "union", "enum", "namespace"}

QUAL_AFTER_PARAMS = {
    # tokens that may appear between ')' and '{' in function defs
    "const", "constexpr", "noexcept", "override", "final", "volatile", "&", "&&", "mutable",
    "requires", "->", "throw",
}

MARKER_START_RE = re.compile(r"^\s*//\s*\[\[\[\s*REPLACE_START\s*:\s*(.+?)\s*\]\]\]\s*$")
MARKER_END_RE = re.compile(r"^\s*//\s*\[\[\[\s*REPLACE_END\s*\]\]\]\s*$")


@dataclass(frozen=True)
class Tok:
    kind: str  # "id", "kw", "op", "punc"
    text: str
    start: int
    end: int  # exclusive


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def lex_cpp(src: str) -> List[Tok]:
    """
    Very small lexer:
    - Removes comments and literals from tokenization (but keeps positions in original text)
    - Produces identifiers/keywords and operators/punctuation tokens we need
    """
    toks: List[Tok] = []
    i = 0
    n = len(src)

    def add(kind: str, text: str, s: int, e: int):
        toks.append(Tok(kind=kind, text=text, start=s, end=e))

    # handle raw strings: R"( ... )" or R"delim( ... )delim"
    def lex_raw_string(start_i: int) -> int:
        # src[start_i] == 'R' and next is '"'
        j = start_i + 2  # after R"
        # read delimiter until '('
        delim = ""
        while j < n and src[j] != "(":
            delim += src[j]
            j += 1
        if j >= n:
            return n
        # now find )delim"
        end_pat = ")" + delim + '"'
        k = src.find(end_pat, j + 1)
        return n if k == -1 else k + len(end_pat)

    while i < n:
        c = src[i]

        # whitespace
        if c.isspace():
            i += 1
            continue

        # line comment
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            i = src.find("\n", i)
            if i == -1:
                break
            continue

        # block comment
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue

        # string literal
        if c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            i = j
            continue

        # raw string literal
        if c == "R" and i + 1 < n and src[i + 1] == '"':
            i = lex_raw_string(i)
            continue

        # char literal
        if c == "'":
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "'":
                    j += 1
                    break
                j += 1
            i = j
            continue

        # identifier / keyword
        if c.isalpha() or c == "_":
            s = i
            i += 1
            while i < n and _is_ident_char(src[i]):
                i += 1
            text = src[s:i]
            kind = "kw" if text in CPP_KEYWORDS else "id"
            add(kind, text, s, i)
            continue

        # multi-char operators/punct we care about
        two = src[i:i + 2]
        three = src[i:i + 3]

        if three in {"->*", "..."}:
            add("op", three, i, i + 3)
            i += 3
            continue
        if two in {"::", "->", ">>", "<<", "<=", ">=", "==", "!=", "&&", "||", "++", "--"}:
            add("op", two, i, i + 2)
            i += 2
            continue

        # single char punctuation/operators
        if c in "{}()[];:,.<>~*&=+-/!%^|?":
            kind = "punc" if c in "{}()[];:,.<>" else "op"
            add(kind, c, i, i + 1)
            i += 1
            continue

        # fallback: unknown single char
        add("op", c, i, i + 1)
        i += 1

    return toks


def line_start(src: str, pos: int) -> int:
    """Start index of the line containing pos."""
    j = src.rfind("\n", 0, pos)
    return 0 if j == -1 else j + 1


def line_end(src: str, pos: int) -> int:
    """End index (exclusive) of the line containing pos."""
    j = src.find("\n", pos)
    return len(src) if j == -1 else j + 1


def get_line(src: str, pos: int) -> str:
    return src[line_start(src, pos):line_end(src, pos)]


def get_indent_of_line(src: str, pos: int) -> str:
    ln = src[line_start(src, pos):line_end(src, pos)]
    return ln[: len(ln) - len(ln.lstrip(" \t"))]


def _prev_line_start(src: str, cur_line_start: int) -> Optional[int]:
    """Start index of previous line (may be empty), given current line start."""
    if cur_line_start <= 0:
        return None
    j = src.rfind("\n", 0, cur_line_start - 1)
    return 0 if j == -1 else j + 1


def prev_nonempty_line_start(src: str, cur_line_start: int) -> Optional[int]:
    """Start index of previous non-empty line (skips blank/whitespace-only lines)."""
    ls = _prev_line_start(src, cur_line_start)
    while ls is not None:
        le = line_end(src, ls)
        if src[ls:le].strip() != "":
            return ls
        ls = _prev_line_start(src, ls)
    return None


def _next_nonempty_line_start(src: str, cur_line_end: int) -> Optional[int]:
    """Start index of next non-empty line (skips blank/whitespace-only lines)."""
    if cur_line_end >= len(src):
        return None
    ls = cur_line_end
    while ls < len(src):
        le = line_end(src, ls)
        if src[ls:le].strip() != "":
            return ls
        ls = le
    return None


def line_stripped(src: str, line_start_pos: int) -> str:
    return src[line_start_pos:line_end(src, line_start_pos)].strip()


def already_has_start_marker(src: str, start_pos: int) -> bool:
    # Check previous non-empty line for an existing start marker
    if start_pos <= 0:
        return False
    prev_ls = prev_nonempty_line_start(src, start_pos)
    if prev_ls is None:
        return False
    prev_line = src[prev_ls:line_end(src, prev_ls)].strip()
    return bool(MARKER_START_RE.match(prev_line))


def already_has_end_marker(src: str, end_brace_pos: int) -> bool:
    # Check next non-empty line after closing brace line
    after_line_end = line_end(src, end_brace_pos)
    nxt_ls = _next_nonempty_line_start(src, after_line_end)
    if nxt_ls is None:
        return False
    nxt = src[nxt_ls:line_end(src, nxt_ls)].strip()
    return bool(MARKER_END_RE.match(nxt))


def find_matching_brace(toks: List[Tok], lbrace_idx: int) -> Optional[int]:
    depth = 0
    for i in range(lbrace_idx, len(toks)):
        if toks[i].text == "{":
            depth += 1
        elif toks[i].text == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def find_matching_lparen(toks: List[Tok], rparen_idx: int) -> Optional[int]:
    depth = 0
    for i in range(rparen_idx, -1, -1):
        if toks[i].text == ")":
            depth += 1
        elif toks[i].text == "(":
            depth -= 1
            if depth == 0:
                return i
    return None


def tok_is_name_like(t: Tok) -> bool:
    # identifier OR keyword 'operator' OR destructor '~' handled separately
    return (t.kind == "id") or (t.kind == "kw" and t.text == "operator")


def extract_name_before_lparen(src: str, toks: List[Tok], lparen_idx: int) -> Optional[Tuple[str, int, int]]:
    """
    Extract function name (possibly qualified) immediately before '(' at lparen_idx.
    Returns (name_string, name_start_pos, name_end_pos_exclusive_of_name).

    Key rule:
      Expand left ONLY through '::' qualifiers.
      This prevents including return types like 'ge::graphStatus'.
    """
    if lparen_idx <= 0:
        return None

    j = lparen_idx - 1

    # Skip lambda: [...](...) {
    if toks[j].text == "]":
        return None

    # operator overload: operator<<(
    # Tokens: 'operator'(kw) '<<'(op) '('
    if toks[j].kind == "op":
        if j - 1 >= 0 and toks[j - 1].kind == "kw" and toks[j - 1].text == "operator":
            base_start_idx = j - 1
            name_start = toks[base_start_idx].start
            name_end = toks[j].end
            # Now expand left through :: qualifiers (A::operator<<)
            k = base_start_idx
        else:
            return None
    else:
        # Normal identifier or 'operator' keyword before '('
        if not (toks[j].kind == "id" or (toks[j].kind == "kw" and toks[j].text == "operator")):
            # destructor: ~Foo(
            if toks[j].kind == "id" and j - 1 >= 0 and toks[j - 1].text == "~":
                name_start = toks[j - 1].start
                name_end = toks[j].end
                k = j - 1
            else:
                return None
        else:
            # Exclude control keywords like if/for/while/switch/catch
            if toks[j].kind == "kw" and toks[j].text in CONTROL_KEYWORDS:
                return None

            name_end = toks[j].end
            k = j

            # destructor: ~Foo
            if toks[k].kind == "id" and k - 1 >= 0 and toks[k - 1].text == "~":
                k -= 1

            name_start = toks[k].start

    def consume_template_qualifier_end_at(idx: int) -> Optional[int]:
        """
        idx points to the token immediately before '::' (the end of a qualifier segment).
        Return the token index of the START of that qualifier segment.
        Handles:
          - Simple id: Foo
          - Template id: Foo<T, U>
        """
        if idx < 0:
            return None

        # Simple qualifier: identifier
        if toks[idx].kind == "id" or (toks[idx].kind == "kw" and toks[idx].text == "operator"):
            return idx

        # Template qualifier ends at '>' or '>>'
        if toks[idx].text in {">", ">>"}:
            depth = 0
            i = idx
            while i >= 0:
                if toks[i].text == ">>":
                    depth += 2
                elif toks[i].text == ">":
                    depth += 1
                elif toks[i].text == "<":
                    depth -= 1
                    if depth == 0:
                        # token before '<' should be the template name (identifier)
                        if i - 1 >= 0 and toks[i - 1].kind == "id":
                            return i - 1
                        return None
                i -= 1
            return None

        return None

    # Expand left ONLY through :: chains (qualified names)
    # pattern: <qualifier> :: <current...>
    while k - 1 >= 0 and toks[k - 1].text == "::":
        # include ::
        k -= 1
        # include qualifier segment to the left of ::
        qual_end = k - 1
        qual_start = consume_template_qualifier_end_at(qual_end)
        if qual_start is None:
            break
        k = qual_start
        name_start = toks[k].start

    name = src[name_start:name_end].strip()
    if not name:
        return None

    return (name, name_start, name_end)


def _is_control_paren(toks: List[Tok], lparen_idx: int) -> bool:
    return lparen_idx > 0 and toks[lparen_idx - 1].kind == "kw" and toks[lparen_idx - 1].text in CONTROL_KEYWORDS


def _is_lambda_param_list(toks: List[Tok], lparen_idx: int) -> bool:
    # [...](args) {  -> token right before '(' is ']'
    return lparen_idx > 0 and toks[lparen_idx - 1].text == "]"


def find_param_rparen_for_function(src: str, toks: List[Tok], lbrace_idx: int) -> Optional[int]:
    """
    For a '{' token at lbrace_idx, search backwards for the parameter-list ')'
    that belongs to the function signature, not to constructor initializers, etc.

    Critical rule:
      Do NOT "jump" from an inner block '{' (if/for/while/lambda/scope) to an outer function signature.
      If the nearest (...) before this '{' is control flow or lambda params, treat as NOT a function.
      If we see another '{' before any viable ')', treat as nested scope and stop.
    """
    i = lbrace_idx - 1
    while i >= 0:
        t = toks[i]

        # statement boundary: cannot be a function definition
        if t.text in {";", "}"}:
            return None

        # nested scope boundary: prevents binding to outer signatures
        if t.text == "{":
            return None

        if t.text == ")":
            lparen_idx = find_matching_lparen(toks, i)
            if lparen_idx is None:
                i -= 1
                continue

            # If nearest (...) is control-flow, this '{' is control block, not function body.
            if _is_control_paren(toks, lparen_idx):
                return None

            # If nearest (...) belongs to a lambda, this '{' is lambda body, not function body.
            if _is_lambda_param_list(toks, lparen_idx):
                return None

            # name must be right before '('
            name_info = extract_name_before_lparen(src, toks, lparen_idx)
            if name_info is None:
                i -= 1
                continue

            # Constructor initializer list filter:
            # If token immediately before the name (skipping '::' chains) is ':' or ',' then it's likely an initializer call.
            # (This blocks 'a(x)' inside ': a(x), b(y) {')
            name, name_start, _ = name_info
            name_tok_idx = None
            for back in range(lparen_idx - 1, -1, -1):
                if toks[back].start == name_start:
                    name_tok_idx = back
                    break
                if toks[back].start < name_start:
                    break
            if name_tok_idx is not None and name_tok_idx - 1 >= 0:
                prev_tok = toks[name_tok_idx - 1]
                if prev_tok.text in {":", ","}:
                    i -= 1
                    continue

            # Ensure between this ')' and '{' we don't see something that disqualifies it
            # like '=' at top-level (lambda init) or ';' (statement end).
            j = i + 1
            ok = True
            while j < lbrace_idx:
                tt = toks[j].text
                if tt in {"=", ";"}:
                    ok = False
                    break
                j += 1
            if not ok:
                i -= 1
                continue

            return i

        i -= 1

    return None


# def compute_signature_start(src: str, name_pos: int) -> int:
#     """
#     Compute where to insert the START marker:
#     - Start at the line containing the function name (or return type)
#     - If immediately preceding non-empty lines are 'template ...' or attributes, include them.
#     """
#     cur = line_start(src, name_pos)

#     while True:
#         prev_ls = prev_nonempty_line_start(src, cur)
#         if prev_ls is None:
#             break

#         prev_line = src[prev_ls:cur]
#         stripped = prev_line.strip()

#         if stripped == "":
#             break

#         ls = prev_line.lstrip()
#         if ls.startswith("template") or ls.startswith("[[") or ls.startswith("__attribute__"):
#             cur = prev_ls
#             continue

#         break

#     return cur


# def add_markers_to_source(src: str) -> Tuple[str, int]:
#     toks = lex_cpp(src)
#     insertions: List[Tuple[int, str]] = []  # (pos, text)

#     # 使用集合记录已经处理的函数开始位置，避免重复
#     processed_starts = set()

#     i = 0
#     while i < len(toks):
#         if toks[i].text != "{":
#             i += 1
#             continue

#         # ... [前面的检查代码保持不变，包括类型块检查等] ...

#         # Skip if already marked
#         if already_has_start_marker(src, sig_start_pos) or already_has_end_marker(src, rbrace_pos):
#             i += 1
#             continue

#         # 防止重复处理同一个函数
#         if sig_start_pos in processed_starts:
#             i += 1
#             continue
#         processed_starts.add(sig_start_pos)

#         # 获取标记起始行的缩进
#         marker_line_start = line_start(src, sig_start_pos)
#         marker_line_end = line_end(src, marker_line_start)
#         marker_line = src[marker_line_start:marker_line_end]
#         indent_start = marker_line[:len(marker_line) - len(marker_line.lstrip())]

#         # 结束标记使用函数体结束行的缩进
#         indent_end = get_indent_of_line(src, rbrace_pos)

#         start_marker = f"{indent_start}// [[[ REPLACE_START: {func_name} ]]]\n"
        
#         # End marker: place it on the next line after the closing brace line
#         rb_end = toks[rbrace_idx].end
#         if rb_end < len(src) and src[rb_end:rb_end + 1] == "\n":
#             end_insert_pos = rb_end + 1
#             end_marker = f"{indent_end}// [[[ REPLACE_END ]]]\n"
#         else:
#             end_insert_pos = rb_end
#             end_marker = f"\n{indent_end}// [[[ REPLACE_END ]]]\n"

#         insertions.append((sig_start_pos, start_marker))
#         insertions.append((end_insert_pos, end_marker))

#         i += 1

#     if not insertions:
#         return src, 0

#     # 对插入位置进行排序并累加偏移量
#     insertions.sort(key=lambda x: x[0])
#     out = src
#     cumulative_shift = 0
    
#     for pos, txt in insertions:
#         actual_pos = pos + cumulative_shift
#         out = out[:actual_pos] + txt + out[actual_pos:]
#         cumulative_shift += len(txt)

#     return out, len(insertions) // 2
def compute_signature_start(src: str, name_pos: int) -> int:
    """
    Compute where to insert the START marker:
    - Start at the line containing the function name (or return type)
    - We want to EXCLUDE template declarations, but include any attributes.
    - The marker should be placed right before the function signature line.
    """
    # First, find the line containing the function name
    name_line_start = line_start(src, name_pos)
    
    # Look backwards to find the beginning of the function signature
    # We want to include any lines that are part of the signature (e.g., return type on previous line)
    # But we want to EXCLUDE template lines and standalone comments.
    cur = name_line_start
    
    while True:
        prev_ls = prev_nonempty_line_start(src, cur)
        if prev_ls is None:
            break
            
        prev_line = src[prev_ls:line_end(src, prev_ls)]
        stripped = prev_line.strip()
        
        # If it's a template declaration, STOP here - don't include it
        if stripped.startswith("template"):
            break
            
        # If it's an attribute ([[...]]), include it
        if stripped.startswith("[["):
            cur = prev_ls
            continue
            
        # If it's a comment line, break - we don't want to include comments in the signature
        if stripped.startswith("//"):
            break
            
        # Check if this line could be part of the function return type
        # Look for keywords that often appear in function signatures
        sig_keywords = {"inline", "static", "virtual", "explicit", "constexpr", "auto", "void", "int", 
                       "bool", "char", "float", "double", "long", "short", "unsigned", "signed",
                       "class", "typename", "typename", "const", "volatile", "noexcept", "override",
                       "final", "friend", "public", "protected", "private"}
        
        # Split the line into words and check for signature keywords
        words = stripped.split()
        has_sig_keyword = any(word in sig_keywords for word in words)
        
        # Also check for common patterns in return types
        # If the line ends with '&', '*', '&&', or contains "::", it might be part of return type
        line_text = stripped.rstrip()
        line_end_chars = line_text.endswith(('&', '*', '&&', 'const', 'volatile', 'noexcept'))
        has_namespace = "::" in stripped
        has_type_like = any(word[0].isupper() for word in words if word)  # crude heuristic for type names
        
        # Check for function-like patterns: contains '(' or ')'
        has_paren = '(' in stripped or ')' in stripped
        
        if has_sig_keyword or line_end_chars or has_namespace or has_type_like or has_paren:
            # This line is likely part of the function signature
            cur = prev_ls
            continue
        else:
            # Not part of function signature, stop here
            break
    
    return cur


def add_markers_to_source(src: str) -> Tuple[str, int]:
    toks = lex_cpp(src)
    insertions: List[Tuple[int, str]] = []  # (pos, text)

    # 使用集合记录已经处理的函数开始位置，避免重复
    processed_starts = set()

    i = 0
    while i < len(toks):
        if toks[i].text != "{":
            i += 1
            continue

        # Exclude type blocks quickly
        # Look back a few tokens for a keyword just before '{' (e.g., "namespace x {")
        is_type_block = False
        for back in range(1, 6):
            if i - back < 0:
                break
            if toks[i - back].kind == "kw" and toks[i - back].text in TYPE_BLOCK_KEYWORDS:
                is_type_block = True
                break
        if is_type_block:
            i += 1
            continue

        # Try detect function signature for this '{'
        rparen_idx = find_param_rparen_for_function(src, toks, i)
        if rparen_idx is None:
            i += 1
            continue

        lparen_idx = find_matching_lparen(toks, rparen_idx)
        if lparen_idx is None:
            i += 1
            continue

        name_info = extract_name_before_lparen(src, toks, lparen_idx)
        if name_info is None:
            i += 1
            continue
        func_name, name_start_pos, _ = name_info

        # Find matching closing brace for this function body
        rbrace_idx = find_matching_brace(toks, i)
        if rbrace_idx is None:
            i += 1
            continue

        rbrace_pos = toks[rbrace_idx].start

        # Determine insertion positions
        sig_start_pos = compute_signature_start(src, name_start_pos)
        
        # Skip if already marked
        if already_has_start_marker(src, sig_start_pos) or already_has_end_marker(src, rbrace_pos):
            i += 1
            continue

        # 防止重复处理同一个函数
        if sig_start_pos in processed_starts:
            i += 1
            continue
        processed_starts.add(sig_start_pos)

        # 获取标记起始行的缩进
        marker_line_start = line_start(src, sig_start_pos)
        marker_line_end = line_end(src, marker_line_start)
        marker_line = src[marker_line_start:marker_line_end]
        indent_start = marker_line[:len(marker_line) - len(marker_line.lstrip())]

        # 结束标记使用函数体结束行的缩进
        indent_end = get_indent_of_line(src, rbrace_pos)

        start_marker = f"{indent_start}// [[[ REPLACE_START: {func_name} ]]]\n"
        
        # End marker: place it on the next line after the closing brace line
        rb_end = toks[rbrace_idx].end
        if rb_end < len(src) and src[rb_end:rb_end + 1] == "\n":
            end_insert_pos = rb_end + 1
            end_marker = f"{indent_end}// [[[ REPLACE_END ]]]\n"
        else:
            end_insert_pos = rb_end
            end_marker = f"\n{indent_end}// [[[ REPLACE_END ]]]\n"

        insertions.append((sig_start_pos, start_marker))
        insertions.append((end_insert_pos, end_marker))

        i += 1

    if not insertions:
        return src, 0

    # 对插入位置进行排序并累加偏移量
    insertions.sort(key=lambda x: x[0])
    out = src
    cumulative_shift = 0
    
    for pos, txt in insertions:
        actual_pos = pos + cumulative_shift
        out = out[:actual_pos] + txt + out[actual_pos:]
        cumulative_shift += len(txt)

    return out, len(insertions) // 2

def iter_cpp_files(paths: List[str], recursive: bool) -> List[str]:
    exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}
    out: List[str] = []

    for p in paths:
        if os.path.isdir(p):
            if not recursive:
                continue
            for root, _, files in os.walk(p):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in exts:
                        out.append(os.path.join(root, fn))
        else:
            if os.path.splitext(p)[1].lower() in exts:
                out.append(p)

    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Add [[[ REPLACE_START/END ]]] markers around C++ function definitions.")
    ap.add_argument("paths", nargs="+", help="Input file(s) or directory(ies).")
    ap.add_argument("-i", "--inplace", action="store_true", help="Modify files in-place.")
    ap.add_argument("-o", "--output", help="Write output to a single file (only valid with one input file).")
    ap.add_argument("--recursive", action="store_true", help="Recurse into directories.")
    ap.add_argument("--backup", action="store_true", help="When --inplace, create .bak backup files.")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes; just report what would change.")
    args = ap.parse_args()

    files = iter_cpp_files(args.paths, recursive=args.recursive)

    if args.output and len(files) != 1:
        ap.error("--output requires exactly one input file")

    total_funcs = 0
    changed_files = 0

    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()

        new_src, func_count = add_markers_to_source(src)
        if func_count == 0 or new_src == src:
            continue

        total_funcs += func_count
        changed_files += 1

        if args.dry_run:
            print(f"[DRY] {fp}: would mark {func_count} function(s)")
            continue

        if args.output:
            with open(args.output, "w", encoding="utf-8") as wf:
                wf.write(new_src)
            print(f"[OK] wrote {args.output}: marked {func_count} function(s)")
            return 0

        if args.inplace:
            if args.backup:
                bak = fp + ".bak"
                with open(bak, "w", encoding="utf-8") as bf:
                    bf.write(src)
            with open(fp, "w", encoding="utf-8") as wf:
                wf.write(new_src)
            print(f"[OK] {fp}: marked {func_count} function(s)")
        else:
            # default: print to stdout (safe when running on one file)
            print(new_src)
            return 0

    if args.dry_run or args.inplace:
        print(f"Done. Changed {changed_files} file(s), marked {total_funcs} function(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
