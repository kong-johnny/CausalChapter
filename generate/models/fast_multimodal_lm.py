# -*- coding: utf-8 -*-
"""Memory-efficient extensions for the existing MultiModalCausalLM wrapper."""
import torch

from .multimodal_lm import MultiModalCausalLM


class FastMultiModalCausalLM(MultiModalCausalLM):
    """Adds targeted-logit forward without changing training/generation APIs."""

    def _multimodal_inputs_embeds(
        self, input_ids, vis_seq=None, vis_mask=None
    ):
        embeds = self.llm.get_input_embeddings()(input_ids).clone()
        embeds = embeds.to(dtype=next(self.llm.parameters()).dtype)
        if vis_seq is None or vis_mask is None or self.disable_visual:
            return embeds
        if self.fusion_mode == "text_only":
            for token_id in self.vis_token_ids:
                embeds[input_ids == token_id] = 0.0
        elif self.fusion_mode == "vis_only":
            embeds = self._forward_vis_only(input_ids, embeds, vis_seq, vis_mask)
        elif self.fusion_mode == "linear":
            embeds = self._forward_linear(input_ids, embeds, vis_seq, vis_mask)
        elif self.fusion_mode == "moe":
            embeds = self._forward_moe(input_ids, embeds, vis_seq, vis_mask)
        else:
            embeds, _ = self._inject_visual_embeds(
                input_ids, embeds, vis_seq, vis_mask
            )
        return embeds

    @torch.no_grad()
    def forward_logits_at_positions(
        self,
        input_ids,
        attention_mask,
        logit_positions,
        vis_seq=None,
        vis_mask=None,
    ):
        """Return vocabulary logits only at requested causal positions.

        `logit_positions[b, t]` identifies the hidden-state position whose
        logits predict target token `t`. Running the decoder backbone first and
        applying the LM head only after gathering these positions avoids a
        `[batch, full_sequence, vocabulary]` allocation.
        """
        embeds = self._multimodal_inputs_embeds(
            input_ids, vis_seq=vis_seq, vis_mask=vis_mask
        )
        causal_lm = (
            self.llm.get_base_model()
            if hasattr(self.llm, "get_base_model")
            else self.llm
        )
        backbone = getattr(causal_lm, "model", None)
        if backbone is None:
            raise TypeError(
                "Targeted KL forward requires a causal LM exposing `.model` "
                "and `.get_output_embeddings()` (supported by LLaMA/Qwen)."
            )
        outputs = backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        # Cross-version compat: try last_hidden_state, fallback to hidden_states[-1]
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        else:
            raise AttributeError(
                "Backbone output has neither 'last_hidden_state' nor 'hidden_states'. "
                "Ensure output_hidden_states=True is passed to the backbone call."
            )
        if logit_positions.ndim != 2:
            raise ValueError("logit_positions must have shape [batch, target_len]")
        gather_index = logit_positions.to(hidden.device).unsqueeze(-1)
        gather_index = gather_index.expand(-1, -1, hidden.shape[-1])
        selected_hidden = torch.gather(hidden, dim=1, index=gather_index)
        lm_head = causal_lm.get_output_embeddings()
        return lm_head(selected_hidden)
