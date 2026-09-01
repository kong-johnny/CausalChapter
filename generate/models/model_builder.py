# -*- coding: utf-8 -*-
"""LoRA/base model construction for the CausalChapter pipeline.

Supports text_only mode (no visual features, no Q-Former) as the default
for Qwen2.5-3B pure-text backbone.
"""
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import prepare_model_for_kbit_training
except ImportError:
    prepare_model_for_kbit_training = None

from .qformer import SimpleQFormer


def clone_base_model_for_adapter(
    base_model_path,
    tokenizer,
    gradient_checkpointing=False,
    use_4bit=False,
    use_8bit=False,
    attn_implementation="sdpa",
):
    if use_4bit and use_8bit:
        raise ValueError("Cannot enable both 4-bit and 8-bit quantization")
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": False,
    }

    if attn_implementation == "auto":
        try:
            import flash_attn  # noqa: F401
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                attn_implementation = "flash_attention_2"
            else:
                attn_implementation = "sdpa"
        except ImportError:
            attn_implementation = "sdpa"
    load_kwargs["attn_implementation"] = attn_implementation
    print(f"[ATTN] requested={attn_implementation}")

    if use_4bit or use_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            load_in_8bit=use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = (
            {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    print(
        "[ATTN] actual=",
        getattr(model.config, "_attn_implementation", "unknown"),
    )

    if use_4bit or use_8bit:
        if prepare_model_for_kbit_training is None:
            raise ImportError("PEFT prepare_model_for_kbit_training is unavailable")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def build_multimodal_model(
    base_model,
    base_model_name,
    tokenizer,
    lora_r=64,
    lora_alpha=128,
    gradient_checkpointing=False,
    use_4bit=False,
    use_8bit=False,
    fusion_mode="text_only",
    num_query_tokens=4,
    vis_hidden=512,
    linear_fusion_type="add",
    moe_num_experts=3,
    moe_top_k=2,
    attn_implementation="sdpa",
):
    from .fast_multimodal_lm import FastMultiModalCausalLM

    llm = clone_base_model_for_adapter(
        base_model,
        tokenizer,
        gradient_checkpointing=gradient_checkpointing,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        attn_implementation=attn_implementation,
    )
    llm = get_peft_model(
        llm,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            base_model_name_or_path=base_model_name,
        ),
    )
    hidden = llm.get_input_embeddings().weight.shape[1]

    # ---- text_only mode: no Q-Former, no VIS tokens ----
    if fusion_mode == "text_only":
        num_query_tokens = 0
        vis_token_ids = []
        qformer = None
        print("[BUILD] text_only mode: Q-Former disabled, no visual features")
    else:
        vis_token_ids = tokenizer.convert_tokens_to_ids(
            [f"<VIS_{i}>" for i in range(num_query_tokens)]
        )
        qformer = SimpleQFormer(
            vis_dim=vis_hidden,
            q_hidden=512,
            llm_hidden=hidden,
            num_query_tokens=num_query_tokens,
            num_heads=8,
            num_layers=2,
        ).to(llm.device if hasattr(llm, "device") else "cpu")
        print(f"[BUILD] fusion_mode={fusion_mode}, num_query_tokens={num_query_tokens}")

    model = FastMultiModalCausalLM(
        llm=llm,
        qformer=qformer,
        vis_token_ids=vis_token_ids,
        use_quantization=(use_4bit or use_8bit),
        fusion_mode=fusion_mode,
        llm_hidden=hidden,
        vis_hidden=vis_hidden,
        num_query_tokens=num_query_tokens,
        linear_fusion_type=linear_fusion_type,
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
    )
    # In text_only mode, disable visual injection entirely
    if fusion_mode == "text_only":
        model.disable_visual = True

    return model, model