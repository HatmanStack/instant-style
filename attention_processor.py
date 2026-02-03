"""Attention processor for IP-Adapter with Flux models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm


class IPAFluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int | None = None,
        scale: float = 1.0,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size  # 3072
        self.cross_attention_dim = cross_attention_dim  # 4096
        self.scale = scale  # Default scale, can be overridden per-call
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.FloatTensor,
        image_emb: torch.FloatTensor | None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.FloatTensor | tuple[torch.FloatTensor, torch.FloatTensor]:
        """Process attention with optional IP-Adapter image embeddings.

        Args:
            attn: The attention module
            hidden_states: Input hidden states
            image_emb: Optional image embeddings from IP-Adapter
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask
            image_rotary_emb: Optional rotary embeddings for images
            mask: Optional mask tensor
            scale: Scale factor for IP-Adapter contribution (defaults to self.scale)

        Returns:
            Output hidden states, or tuple of (hidden_states, encoder_hidden_states)
            when encoder_hidden_states is provided
        """
        # Use provided scale or fall back to instance default
        effective_scale = scale if scale is not None else self.scale

        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        ip_hidden_states: torch.Tensor | None = None
        if image_emb is not None:
            # `ip-adapter` projections
            ip_hidden_states_input = image_emb
            ip_hidden_states_key_proj = self.to_k_ip(ip_hidden_states_input)
            ip_hidden_states_value_proj = self.to_v_ip(ip_hidden_states_input)

            ip_hidden_states_key_proj = ip_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            ip_hidden_states_value_proj = ip_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            ip_hidden_states_key_proj = self.norm_added_k(ip_hidden_states_key_proj)

            ip_hidden_states = F.scaled_dot_product_attention(
                query,
                ip_hidden_states_key_proj,
                ip_hidden_states_value_proj,
                dropout_p=0.0,
                is_causal=False,
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states_out, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            if ip_hidden_states is not None:
                hidden_states = hidden_states + effective_scale * ip_hidden_states

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

            return hidden_states, encoder_hidden_states_out
        else:
            if ip_hidden_states is not None:
                hidden_states = hidden_states + effective_scale * ip_hidden_states

            return hidden_states
