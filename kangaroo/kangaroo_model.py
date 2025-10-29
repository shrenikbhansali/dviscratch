"""Kangaroo model wrapper with optional identity adapter construction."""
from __future__ import annotations

import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from fastchat.utils import str_to_torch_dtype
from transformers import AutoConfig
from transformers.models.llama import LlamaConfig

from kangaroo.adapter import (
    AdapterModel,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    _expand_mask,
    _make_causal_mask,
)
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.drafter_head import DrafterHead


class IdentityDecoderLayer(nn.Module):
    """Decoder layer mirroring the base model's attention and MLP stack."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: Tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class IdentityAdapterModel(nn.Module):
    """Adapter built directly from the base model's shallow layers."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([IdentityDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> Optional[torch.Tensor]:
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None,
    ):
        return self.forward_early_stop(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            std,
        )

    def forward_early_stop(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None,
    ):
        if inputs_embeds is None:
            raise ValueError("IdentityAdapterModel expects inputs_embeds")

        batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                cache_entry = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache.append(cache_entry)

        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(next_decoder_cache)

        return hidden_states


class KangarooModel(nn.Module):
    """Wrapper that owns the verifier backbone and shallow adapter."""

    def __init__(
        self,
        *,
        model_id: str,
        adapter_mode: str,
        adapter_path: Optional[str],
        exit_layer: int,
        dtype: str = "float16",
        device_map: Optional[str] = "auto",
        **unused_kwargs,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.adapter_mode = adapter_mode
        self.adapter_path = adapter_path
        self.exit_layer = exit_layer

        self.base_config = AutoConfig.from_pretrained(model_id)
        self._validate_exit_layer()

        torch_dtype = str_to_torch_dtype(dtype) if dtype else None
        self.base_model = EarlyExitLlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            EARLY_STOP_LAYER=exit_layer,
        )
        self.base_model = self.base_model.eval()
        self.base_config = self.base_model.config

        self.head_model = self._resolve_head_model()

        normalized_path = adapter_path
        if adapter_mode == "load":
            self._load_adapter_from_path(normalized_path, exit_layer=exit_layer)
        else:
            self._build_identity_adapter_from_base(self.base_config, exit_layer=exit_layer)

    def _resolve_head_model(self) -> nn.Module:
        head = getattr(self.base_model, "lm_head", None)
        if head is None and hasattr(self.base_model, "get_output_embeddings"):
            head = self.base_model.get_output_embeddings()
        if head is None:
            raise ValueError("KangarooModel requires the base model to expose an output head.")
        head = head.eval()
        target_device = self._first_parameter_device(self.base_model.model.layers[0])
        target_dtype = self._first_parameter_dtype(self.base_model.model.layers[0])
        if target_device is not None or target_dtype is not None:
            head = head.to(device=target_device, dtype=target_dtype)
        return head

    def _load_adapter_from_path(self, adapter_path: Optional[str], *, exit_layer: int) -> None:
        if not adapter_path:
            raise ValueError("adapter-mode=load requires --adapter-path")
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")
        config = LlamaConfig.from_pretrained(adapter_path)
        if exit_layer > config.num_hidden_layers:
            raise ValueError(
                f"exit_layer={exit_layer} exceeds adapter depth {config.num_hidden_layers}"
            )
        config.exit_layer = exit_layer
        config.is_identity_adapter = False
        self.adapter_cfg = config
        self.adapter_model = AdapterModel(config)
        state_path = os.path.join(adapter_path, "pytorch_model.bin")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"missing adapter weights: {state_path}")
        state_dict = torch.load(state_path, map_location="cpu")
        self.adapter_model.load_state_dict(state_dict, strict=False)
        self._finalize_adapter()

    def _build_identity_adapter_from_base(self, base_cfg: LlamaConfig, *, exit_layer: int) -> None:
        """Build an identity adapter by cloning the shallow base layers."""
        if exit_layer <= 0:
            raise ValueError("exit_layer must be positive")
        if exit_layer > len(self.base_model.model.layers):
            raise ValueError(
                f"exit_layer={exit_layer} exceeds base depth {len(self.base_model.model.layers)}"
            )

        adapter_cfg = copy.deepcopy(base_cfg)
        adapter_cfg.num_hidden_layers = exit_layer
        adapter_cfg.is_identity_adapter = True
        adapter_cfg.exit_layer = exit_layer
        self.adapter_cfg = adapter_cfg
        self.adapter_model = IdentityAdapterModel(adapter_cfg)

        for idx in range(exit_layer):
            self.adapter_model.layers[idx].load_state_dict(
                self.base_model.model.layers[idx].state_dict()
            )
        self.adapter_model.norm.load_state_dict(self.base_model.model.norm.state_dict())
        self._finalize_adapter()

    def _finalize_adapter(self) -> None:
        target_device = self._first_parameter_device(self.base_model.model.layers[0])
        target_dtype = self._first_parameter_dtype(self.base_model.model.layers[0])
        if target_device is not None or target_dtype is not None:
            self.adapter_model = self.adapter_model.to(device=target_device, dtype=target_dtype)
        self.adapter_model = self.adapter_model.eval()
        self._freeze_module(self.adapter_model)

    def _freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)

    def _validate_exit_layer(self) -> None:
        total_layers = getattr(self.base_config, "num_hidden_layers", None)
        if total_layers is None:
            raise ValueError("Base model config is missing num_hidden_layers")
        if not (0 < self.exit_layer <= total_layers):
            raise ValueError(
                f"exit_layer must be within [1, {total_layers}] (got {self.exit_layer})"
            )

    @staticmethod
    def _first_parameter_device(module: nn.Module) -> Optional[torch.device]:
        for param in module.parameters():
            return param.device
        return None

    @staticmethod
    def _first_parameter_dtype(module: nn.Module) -> Optional[torch.dtype]:
        for param in module.parameters():
            return param.dtype
        return None

    def forward(self):
        raise NotImplementedError

    def attach_drafter_head(self, *, r: int, alpha: float) -> None:
        """Attach a drafter head cloned from ``self.head_model`` with frozen base weights."""

        if hasattr(self, "drafter_head"):
            return

        base_weight = getattr(self.head_model, "weight", None)
        if base_weight is None:
            raise AttributeError("head_model must expose a weight tensor for cloning")

        vocab_size, hidden_size = base_weight.shape
        bias_tensor = getattr(self.head_model, "bias", None)
        use_bias = bias_tensor is not None

        drafter = DrafterHead(hidden_size, vocab_size, r=r, alpha=alpha, bias=use_bias)

        target_device = base_weight.device
        target_dtype = base_weight.dtype
        drafter = drafter.to(device=target_device, dtype=target_dtype)

        drafter.proj.base.weight.data.copy_(base_weight.data)
        drafter.proj.base.weight.requires_grad_(False)

        if use_bias:
            assert drafter.proj.base.bias is not None
            drafter.proj.base.bias.data.copy_(bias_tensor.data)
            drafter.proj.base.bias.requires_grad_(False)

        self.drafter_head = drafter

    def drafter_logits_from_hk(self, hk: "torch.Tensor") -> "torch.Tensor":
        """Return drafter logits for hidden states ``hk``; requires ``attach_drafter_head`` first."""

        if not hasattr(self, "drafter_head"):
            raise RuntimeError("drafter_head is not attached; call attach_drafter_head() first.")

        base_weight = self.head_model.weight
        hidden_size = base_weight.shape[1]
        if hk.shape[-1] != hidden_size:
            raise ValueError(
                f"hk hidden size mismatch: expected {hidden_size}, got {hk.shape[-1]}"
            )

        drafter_weight = self.drafter_head.proj.base.weight
        if hk.device != drafter_weight.device or hk.dtype != drafter_weight.dtype:
            raise ValueError(
                "hk tensor device/dtype mismatch with drafter head: "
                f"hk device={hk.device}, dtype={hk.dtype}; "
                f"expected device={drafter_weight.device}, dtype={drafter_weight.dtype}"
            )

        return self.drafter_head(hk)

    def dvi_trainable_params(self) -> "List[nn.Parameter]":
        """Return the LoRA trainable parameters of the drafter head, if present."""

        if not hasattr(self, "drafter_head"):
            return []
        return list(self.drafter_head.lora_params())
