# vdahm.py
from functools import partial
import torch
import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import *
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv

from utils import get_layers
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.cache_utils import Cache
import matplotlib.pyplot as plt
import numpy as np
ALL_SCORES = []

# Adapted from Qwen2Attention.forward
def vdahm_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    aug_ratio: float = 1.0,
    filter: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2_5_VLModel is using Qwen2_5_VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    now_attn_output = attn_output[0, :, -1, :]
    if attn_output.shape[-2] > 1 and (not hasattr(self, "attn_output_tc") or self.attn_output_tc is None):
        self.attn_output_tc = now_attn_output
    
    elif attn_output.shape[-2] > 1:
        head_vary = ((self.attn_output_tc - now_attn_output)**2).sum(-1)
        if head_vary.std() > 1 and filter: 
            aotc = (self.attn_output_tc**2).sum(-1)
            head_vary[(head_vary>(head_vary.mean()+head_vary.std()))&(aotc>(aotc.mean()+aotc.std()))] = 0
        ALL_SCORES.append(head_vary.detach().cpu().to(torch.float32).numpy())
        
        self.attn_output_tc = None
        self.aug_heads = (head_vary > head_vary.median()).nonzero().flatten()
        attn_output[:, self.aug_heads, :, :] = aug_ratio * attn_output[:, self.aug_heads, :, :]

    elif attn_output.shape[-2] == 1:
        attn_output[:, self.aug_heads, :, :] = aug_ratio * attn_output[:, self.aug_heads, :, :]

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def replace_attn(model, target_layers=None, aug_ratio=2, filter=False):
    target_layers = target_layers or []
    
    forward = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLSdpaAttention.forward

    ALL_SCORES = []
    
    for idx, layer in enumerate(get_layers(model)):
        if idx in target_layers:
            layer.self_attn.forward = partial(vdahm_forward, layer.self_attn, aug_ratio=aug_ratio, filter=filter)
        else:
            layer.self_attn.forward = partial(forward, layer.self_attn)