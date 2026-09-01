# -*- coding: utf-8 -*-
"""Unified chat template utilities for training and inference"""


def sharegpt_to_openai(messages):
    """
    将 ShareGPT 格式转换为 OpenAI 格式
    Input: [{"from": "human", "value": "..."}, ...]
    Output: [{"role": "user", "content": "..."}, ...]
    """
    role_map = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
    }
    openai_messages = []
    for msg in messages:
        role = role_map.get(msg["from"])
        if role:
            openai_messages.append({"role": role, "content": msg["value"]})
    return openai_messages


def apply_chat_template_for_training(tokenizer, messages, max_length):
    """
    训练场景：构造 input_ids 和 labels
    1. 渲染 System + User 部分 (带 generation_prompt)
    2. 让 Chat Template 处理 assistant 消息和 EOS
    3. Mask 掉 System+User 部分 (设为 -100)
    """
    # 1. 分离 Prompt 和 Answer
    prompt_msgs = [m for m in messages if m["from"] in ["system", "human"]]
    gpt_msgs = [m for m in messages if m["from"] == "gpt"]

    if len(gpt_msgs) == 0:
        raise ValueError("Training data must contain at least one 'gpt' message.")

    answer_text = gpt_msgs[-1]["value"]

    # 2. 渲染 Prompt 部分 (带 generation_prompt)
    openai_prompt = sharegpt_to_openai(prompt_msgs)
    prompt_text = tokenizer.apply_chat_template(
        openai_prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    # 3. 构建完整对话：prompt + assistant 回复
    # 让 tokenizer 自己处理 assistant 标签和 EOS
    full_msgs = openai_prompt + [{"role": "assistant", "content": answer_text}]

    # 4. Tokenize 完整对话 (包含 prompt + assistant)
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    # 5. 关键修复：找到 assistant generation prompt 的起始位置作为 mask 分界点
    # prompt_text 末尾是 "...<|im_end|>\n<|im_start|>assistant\n"
    # 推理时 prompt 已包含 "<|im_start|>assistant\n"，label 应从其之后开始（答卷内容）
    gen_marker = "<|im_start|>assistant\n"
    if gen_marker in full_text:
        # prompt_part = text BEFORE marker + marker itself
        # 这样 marker 属于 prompt_ids，label 从 marker 之后开始
        before_marker, after_marker = full_text.split(gen_marker, 1)
        prompt_part = before_marker + gen_marker
        answer_part = after_marker  # "<|im_end|>\n" + answer_text + eos
    else:
        raise ValueError(f"gen_marker {repr(gen_marker)} not found in full_text!")

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_part, add_special_tokens=False)["input_ids"]

    # Verify: prompt_ids should be prefix of full_ids
    assert full_ids[:len(prompt_ids)] == prompt_ids, \
        f"Tokenization mismatch!\nPROMPT: {len(prompt_ids)} toks\nFULL: ...{len(full_ids)} toks"

    # prompt_len 是 prompt 部分（不包含 assistant 回答）的 token 数
    prompt_len = len(prompt_ids)
    answer_len = len(answer_ids)

    # 6. 截断逻辑：优先保证 Answer 完整
    if len(full_ids) > max_length:
        if answer_len > max_length:
            # answer 太长，截断 answer
            # 需要重新 tokenize 截断后的 text
            # 取 answer 的最后 max_length tokens
            truncated_answer_ids = answer_ids[-max_length:]
            full_ids = truncated_answer_ids
            prompt_len = 0
        else:
            # 截断 prompt（从头截，保持 answer 完整）
            keep_len = max_length - answer_len
            # prompt_part 的最后 keep_len tokens
            truncated_prompt_ids = prompt_ids[-keep_len:]
            full_ids = truncated_prompt_ids + answer_ids
            prompt_len = keep_len
    else:
        full_ids = prompt_ids + answer_ids

    # 7. 构造 Labels：Prompt 部分全 -100，Answer 部分保留
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return full_ids, labels


def apply_chat_template_for_inference(tokenizer, messages):
    """
    推理场景：构造给模型 generate 的 prompt
    只包含 System + User，末尾带 generation_prompt
    """
    # 过滤掉 assistant 的历史回复（如果是流式推理可能需要保留，这里是单次生成）
    # 对于 Causal Selection 场景，我们只有 System + User
    prompt_msgs = [m for m in messages if m["from"] in ["system", "human"]]

    openai_msgs = sharegpt_to_openai(prompt_msgs)

    prompt_text = tokenizer.apply_chat_template(
        openai_msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt_text