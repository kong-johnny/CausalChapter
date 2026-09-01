# -*- coding: utf-8 -*-
"""Text processing utilities"""
import re


def normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_words(s):
    s = normalize_text(s).lower()
    return re.findall(r"\w+", s)


def text_jaccard(a, b):
    wa = set(tokenize_words(a))
    wb = set(tokenize_words(b))
    if len(wa) == 0 or len(wb) == 0:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union > 0 else 0.0


def safe_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        x = x.strip()
        if ":" in x:
            parts = x.split(":")
            try:
                if len(parts) == 3:
                    h, m, s = parts
                    return float(h) * 3600 + float(m) * 60 + float(s)
                elif len(parts) == 2:
                    m, s = parts
                    return float(m) * 60 + float(s)
            except Exception:
                pass
        try:
            return float(x)
        except Exception:
            return 0.0
    return 0.0


def postprocess_title(title):
    if not title:
        return ""

    # ========== 0. 最先处理：去 special tokens + 取第一行 ==========
    for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "<s>"]:
        title = title.replace(tok, "")

    title = title.strip()
    if "\n" in title:
        title = title.split("\n", 1)[0]

    # ========== 1. 然后再做空白归一化 ==========
    title = normalize_text(title)

    # ========== 2. 兼容旧 prompt 格式残留 marker（仅开头匹配） ==========
    markers = [
        "please generate a concise subheading for the current segment:",
        "human-comprehension summary:",
        "the subheading of the above lecture segment is:",
        "the subheading of the lecture segment is:",
        "the subheading is:",
        "subheading:",
        "human:",
        "assistant:",
    ]
    lower = title.lower()
    for m in markers:
        if lower.startswith(m):
            title = title[len(m):].strip()
            lower = title.lower()

    # ========== 3. 去元信息 ==========
    title = re.sub(r"\[(?:pdf|online|link)\]", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+by\s+[A-Z][A-Za-z.\- ]+$", "", title)

    # ========== 4. 去首尾引号 ==========
    if len(title) >= 2:
        if (title[0] == '"' and title[-1] == '"') or (title[0] == "'" and title[-1] == "'"):
            title = title[1:-1].strip()

    # ========== 5. 去 markdown 残留 ==========
    title = title.strip("*").strip("#").strip("`").strip()

    # ========== 6. 去结尾的不合理标点 ==========
    while len(title) > 0 and title[-1] in [".", ":", ";", ","]:
        title = title[:-1].strip()

    return title


def truncate_text_by_tokens(tokenizer, text, max_tokens):
    """Truncate text by token count"""
    if text is None or not isinstance(text, str) or not text.strip():
        return ""
    ids = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def fit_segment_to_budget(tokenizer, segment_text, prompt_template_fn, max_total_tokens, safety_margin=128):
    """
    动态截断 segment_text，使得用它构造的完整 prompt token 数 ≤ max_total_tokens - safety_margin。
    """
    target = max_total_tokens - safety_margin

    full_prompt = prompt_template_fn(segment_text)
    n_tokens = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])

    if n_tokens <= target:
        return segment_text

    empty_prompt = prompt_template_fn("")
    framework_tokens = len(tokenizer(empty_prompt, add_special_tokens=False)["input_ids"])
    seg_budget = target - framework_tokens

    if seg_budget <= 50:
        seg_budget = 200

    seg_ids = tokenizer(segment_text, add_special_tokens=False)["input_ids"]
    if len(seg_ids) <= seg_budget:
        return segment_text

    # 简单策略：保留前半 + ... + 后半
    head = seg_budget // 2
    tail = seg_budget - head - 10
    truncated_ids = seg_ids[:head] + tokenizer(" ... ", add_special_tokens=False)["input_ids"] + seg_ids[-tail:]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def build_prompt_ids_with_truncation(tokenizer, head, middle, tail, max_total_tokens, safety_margin=128):
    """
    Sandwich truncation returning token IDs (no decode/encode roundtrip).
    Use this for training — the string version is for inference only.

    Returns:
        head_ids: list of int — full head token ids
        middle_ids: list of int — truncated middle token ids
        tail_ids: list of int — tail token ids (always intact)
        budget_used: int — how many tokens were consumed from middle budget
    """
    target = max_total_tokens - safety_margin

    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"] if tail else []
    middle_full_ids = tokenizer(middle, add_special_tokens=False)["input_ids"]

    budget_for_middle = target - len(head_ids) - len(tail_ids)

    if budget_for_middle <= 0:
        raise ValueError(
            f"head({len(head_ids)}) + tail({len(tail_ids)}) = {len(head_ids)+len(tail_ids)} "
            f"exceeds max_total_tokens({max_total_tokens}). Reduce prompt head or increase max_length."
        )

    if len(middle_full_ids) <= budget_for_middle:
        middle_ids = middle_full_ids
        budget_used = len(middle_full_ids)
    else:
        # Left-truncate: keep the END of segment (most recent content)
        middle_ids = middle_full_ids[-budget_for_middle:]
        budget_used = budget_for_middle

    final_ids = head_ids + middle_ids + tail_ids

    # Safety assertion: tail must NOT be corrupted by left-truncation of middle
    if tail_ids:
        assert final_ids[-len(tail_ids):] == tail_ids, \
            "Tail was corrupted during truncation!"

    return final_ids, head_ids, middle_ids, tail_ids, budget_used


def build_prompt_with_truncation(tokenizer, head, middle, tail, max_total_tokens, safety_margin=128):
    """
    Sandwich truncation: keep head and tail intact, truncate middle from the LEFT.

    Args:
        tokenizer: tokenizer
        head: string — beginning of prompt (system + VIS + few-shot + context)
        middle: string — segment text (left-truncate to fit budget)
        tail: string — end of prompt (<|im_end|>\n<|im_start|>assistant\n for inference; empty for training)
        max_total_tokens: total token budget
        safety_margin: buffer for generation output

    Returns:
        final_prompt_text (string)
    """
    target = max_total_tokens - safety_margin

    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"] if tail else []

    middle_full_ids = tokenizer(middle, add_special_tokens=False)["input_ids"]

    budget_for_middle = target - len(head_ids) - len(tail_ids)

    if budget_for_middle <= 0:
        raise ValueError(
            f"head({len(head_ids)}) + tail({len(tail_ids)}) = {len(head_ids)+len(tail_ids)} "
            f"exceeds max_total_tokens({max_total_tokens}). Reduce prompt head or increase max_length."
        )

    if len(middle_full_ids) <= budget_for_middle:
        middle_ids = middle_full_ids
    else:
        # Left-truncate: keep the END of segment (most recent content)
        middle_ids = middle_full_ids[-budget_for_middle:]

    final_ids = head_ids + middle_ids + tail_ids

    # Safety assertion: tail must NOT be corrupted by left-truncation of middle
    if tail_ids:
        assert final_ids[-len(tail_ids):] == tail_ids, \
            "Tail was corrupted during truncation!"

    final_text = tokenizer.decode(final_ids, skip_special_tokens=False)
    return final_text
