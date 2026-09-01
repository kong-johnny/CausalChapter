# -*- coding: utf-8 -*-
"""
Qwen2.5-3B Token Classifier for Video Topic Segmentation

Architecture: Qwen2.5-3B backbone + LoRA + 2-class classification head.
Classification: B-EOP (boundary) / O (non-boundary) at BOS positions only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


class QwenBoundaryClassifier(nn.Module):
    """Qwen2.5-3B with a token classification head for B-EOP/O prediction."""

    def __init__(
        self,
        base_model_name="models/Qwen2.5-3B-Instruct",
        num_labels=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()

        # Load base Qwen2.5 model (bf16 for efficiency)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Extract hidden size
        hidden_size = self.base_model.config.hidden_size

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        print(f"[MODEL] Qwen2.5-3B loaded, LoRA r={lora_r} alpha={lora_alpha}")
        print(f"[MODEL] Trainable params: {sum(p.numel() for p in self.base_model.parameters() if p.requires_grad):,}")

        # 2-class classification head (match model dtype)
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, bos_positions, labels=None):
        """
        Args:
            input_ids: [B, L] tokenized input
            attention_mask: [B, L]
            bos_positions: [B, N] indices of BOS tokens in the sequence
            labels: [B, N] B-EOP/O labels at BOS positions, or None

        Returns:
            logits: [B, N, 2] classification logits at BOS positions
            loss: cross-entropy loss (if labels provided)
        """
        # Get hidden states from Qwen
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # [B, L, H]

        # Extract hidden states at BOS positions
        bos_hidden = self._gather_bos_hidden(hidden_states, bos_positions)  # [B, N, H]

        # Classification
        logits = self.classifier(bos_hidden)  # [B, N, 2]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"logits": logits, "loss": loss}

    def _gather_bos_hidden(self, hidden_states, bos_positions):
        """Extract hidden states at BOS positions.

        Args:
            hidden_states: [B, L, H]
            bos_positions: [B, N] — indices into the sequence dimension

        Returns:
            [B, N, H]
        """
        B, L, H = hidden_states.shape
        B_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand_as(bos_positions)
        return hidden_states[B_idx, bos_positions]

    def inference(self, input_ids, attention_mask, bos_positions):
        """Run inference without gradients. Returns predicted labels."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bos_positions=bos_positions,
                labels=None,
            )
            logits = outputs["logits"]  # [B, N, 2]
            predictions = torch.argmax(logits, dim=-1)  # [B, N]
        return predictions

    def get_llm_for_save(self):
        """Return the underlying PEFT model for saving."""
        return self.base_model