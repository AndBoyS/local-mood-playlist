"""Pure-torch MERT-v1-95M backbone loaded from the vendored asset bundle.

Numeric port of the HuBERT feature-encoder/encoder/encoder-layer stack that
the published 7-layer checkpoint (via upstream's remote-code
``modeling_MERT.py``) targets, valid only for the asset shape asserted by
``_load_config`` (post-LN, plain attention, no CQT, eval-mode masking
no-op). Class structure mirrors the checkpoint's safetensors key tree;
forward math routes through functional calls so ops are spelled out.
fp32 CPU and MPS fp16 outputs are bit-identical to the transformers build
of the same checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.nn import utils as nn_utils

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

# These values also appear inline in the shape annotations below.
MERT_HIDDEN_SIZE: Final = 768
MERT_INTERMEDIATE_SIZE: Final = 3072
MERT_NUM_ATTENTION_HEADS: Final = 12
MERT_ATTENTION_HEAD_DIM: Final = MERT_HIDDEN_SIZE // MERT_NUM_ATTENTION_HEADS
MERT_CONV_DIM: Final = 512
MERT_CONV_DIMS: Final = [MERT_CONV_DIM] * 7
MERT_CONV_KERNELS: Final = [10, 3, 3, 3, 3, 2, 2]
MERT_CONV_STRIDES: Final = [5, 2, 2, 2, 2, 2, 2]
MERT_FEAT_EXTRACT_LAYERS: Final = 7
MERT_HIDDEN_LAYERS: Final = 7
MERT_LAYER_NORM_EPS: Final = 1e-5
MERT_POS_CONV_KERNEL: Final = 128
MERT_POS_CONV_PAD: Final = 64
MERT_POS_CONV_GROUPS: Final = 16
MERT_FEATURE_PROJECTION_DIM: Final = 512
MERT_SAMPLE_RATE: Final = 24000


def mert_sample_rate_check(value: int, *, source: str) -> None:
    """Mirror Wav2Vec2FeatureExtractor's guard: audio must arrive at 24 kHz."""
    if value != MERT_SAMPLE_RATE:
        message = f"{source} requires MERT input sampled at {MERT_SAMPLE_RATE} Hz, got {value}"
        raise ValueError(message)


def _load_config(snapshot: Path) -> None:
    path = snapshot / "config.json"
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        message = f"MERT config must decode to an object: {path}"
        raise TypeError(message)
    expected: dict[str, object] = {
        "conv_dim": MERT_CONV_DIMS,
        "conv_kernel": MERT_CONV_KERNELS,
        "conv_stride": MERT_CONV_STRIDES,
        "num_feat_extract_layers": MERT_FEAT_EXTRACT_LAYERS,
        "num_hidden_layers": MERT_HIDDEN_LAYERS,
        "hidden_size": MERT_HIDDEN_SIZE,
        "intermediate_size": MERT_INTERMEDIATE_SIZE,
        "num_attention_heads": MERT_NUM_ATTENTION_HEADS,
        "feat_extract_norm": "group",
        "conv_bias": False,
        "feat_extract_activation": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "num_conv_pos_embeddings": MERT_POS_CONV_KERNEL,
        "num_conv_pos_embedding_groups": MERT_POS_CONV_GROUPS,
        "feat_proj_layer_norm": True,
        "do_stable_layer_norm": False,
        "feature_extractor_cqt": False,
        "deepnorm": False,
        "attention_relax": -1.0,
        "sample_rate": MERT_SAMPLE_RATE,
    }
    for key, want in expected.items():
        if data.get(key) != want:
            message = f"Unsupported MERT {key}={data.get(key)!r}, expected {want!r} ({snapshot})"
            raise ValueError(message)


class _PosConvEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            MERT_HIDDEN_SIZE,
            MERT_HIDDEN_SIZE,
            kernel_size=MERT_POS_CONV_KERNEL,
            padding=MERT_POS_CONV_PAD,
            groups=MERT_POS_CONV_GROUPS,
        )
        _ = nn_utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

    @override
    def forward[B: IntVar, T: IntVar](self, hidden_states: Tensor[[B, T, 768]]) -> Tensor[[B, T, 768]]:
        reordered = hidden_states.transpose(1, 2)
        projected = F.conv1d(
            reordered,
            self.conv.weight,
            self.conv.bias,
            padding=MERT_POS_CONV_PAD,
            groups=MERT_POS_CONV_GROUPS,
        )
        # kernel 128 is even: HubertSamePadLayer trims exactly one trailing frame.
        trimmed = projected[:, :, :-1]
        activated = F.gelu(trimmed)
        return activated.transpose(1, 2)


class _FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(MERT_HIDDEN_SIZE, MERT_INTERMEDIATE_SIZE)
        self.output_dense = nn.Linear(MERT_INTERMEDIATE_SIZE, MERT_HIDDEN_SIZE)

    @override
    def forward[B: IntVar, T: IntVar](self, hidden_states: Tensor[[B, T, 768]]) -> Tensor[[B, T, 768]]:
        expanded = F.linear(hidden_states, self.intermediate_dense.weight, self.intermediate_dense.bias)
        activated = F.gelu(expanded)
        return F.linear(activated, self.output_dense.weight, self.output_dense.bias)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(MERT_HIDDEN_SIZE, MERT_HIDDEN_SIZE)
        self.k_proj = nn.Linear(MERT_HIDDEN_SIZE, MERT_HIDDEN_SIZE)
        self.v_proj = nn.Linear(MERT_HIDDEN_SIZE, MERT_HIDDEN_SIZE)
        self.out_proj = nn.Linear(MERT_HIDDEN_SIZE, MERT_HIDDEN_SIZE)

    @override
    def forward[B: IntVar, T: IntVar](self, hidden_states: Tensor[[B, T, 768]]) -> Tensor[[B, T, 768]]:
        batch, seq_length, _ = hidden_states.shape
        queries = self.q_proj.forward(hidden_states)
        keys = self.k_proj.forward(hidden_states)
        values = self.v_proj.forward(hidden_states)
        queries = queries.reshape(batch, seq_length, MERT_NUM_ATTENTION_HEADS, MERT_ATTENTION_HEAD_DIM).transpose(1, 2)
        keys = keys.reshape(batch, seq_length, MERT_NUM_ATTENTION_HEADS, MERT_ATTENTION_HEAD_DIM).transpose(1, 2)
        values = values.reshape(batch, seq_length, MERT_NUM_ATTENTION_HEADS, MERT_ATTENTION_HEAD_DIM).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=MERT_ATTENTION_HEAD_DIM**-0.5,
        )
        merged = attn_output.transpose(1, 2).reshape(batch, seq_length, MERT_HIDDEN_SIZE)
        return F.linear(merged, self.out_proj.weight, self.out_proj.bias)


class _EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _Attention()
        self.layer_norm = nn.LayerNorm(MERT_HIDDEN_SIZE, eps=MERT_LAYER_NORM_EPS)
        self.feed_forward = _FeedForward()
        self.final_layer_norm = nn.LayerNorm(MERT_HIDDEN_SIZE, eps=MERT_LAYER_NORM_EPS)

    @override
    def forward[B: IntVar, T: IntVar](self, hidden_states: Tensor[[B, T, 768]]) -> Tensor[[B, T, 768]]:
        attn_output = self.attention.forward(hidden_states)
        normed = self.layer_norm.forward(hidden_states + attn_output)
        fed_forward = self.feed_forward.forward(normed)
        return self.final_layer_norm.forward(normed + fed_forward)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_conv_embed = _PosConvEmbedding()
        self.layer_norm = nn.LayerNorm(MERT_HIDDEN_SIZE, eps=MERT_LAYER_NORM_EPS)
        self.layers = nn.ModuleList(_EncoderLayer() for _ in range(MERT_HIDDEN_LAYERS))

    @override
    def forward[B: IntVar, T: IntVar](
        self, hidden_states: Tensor[[B, T, 768]]
    ) -> tuple[Tensor[[B, T, 768]], tuple[Tensor[[B, T, 768]], ...]]:
        hidden_states = hidden_states + self.pos_conv_embed.forward(hidden_states)
        hidden_states = self.layer_norm.forward(hidden_states)
        captured: list[Tensor[[B, T, 768]]] = []
        for layer in self.layers:
            captured.append(hidden_states)
            hidden_states = layer.forward(hidden_states)
        captured.append(hidden_states)
        return hidden_states, tuple(captured)


class _FeatureProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(MERT_FEATURE_PROJECTION_DIM, eps=MERT_LAYER_NORM_EPS)
        self.projection = nn.Linear(MERT_FEATURE_PROJECTION_DIM, MERT_HIDDEN_SIZE)

    @override
    def forward[B: IntVar, T: IntVar](self, hidden_states: Tensor[[B, T, 512]]) -> Tensor[[B, T, 768]]:
        normed = self.layer_norm.forward(hidden_states)
        return F.linear(normed, self.projection.weight, self.projection.bias)


class _GroupNormConvLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, MERT_CONV_DIM, MERT_CONV_KERNELS[0], stride=MERT_CONV_STRIDES[0], bias=False)
        self.layer_norm = nn.GroupNorm(MERT_CONV_DIM, MERT_CONV_DIM, affine=True)

    @override
    def forward[B: IntVar, T: IntVar](self, hidden_states: Tensor[[B, 1, T]]) -> Tensor[[B, 512, (T - 10) // 5 + 1]]:
        # `(T - 10) // 5 + 1` is not enforced by type checker
        features = F.conv1d(hidden_states, self.conv.weight, bias=None, stride=MERT_CONV_STRIDES[0])
        normalized = F.group_norm(features, MERT_CONV_DIM, self.layer_norm.weight, self.layer_norm.bias)
        return F.gelu(normalized)


class _PlainConvLayer(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            MERT_CONV_DIM,
            MERT_CONV_DIM,
            MERT_CONV_KERNELS[layer_id],
            stride=MERT_CONV_STRIDES[layer_id],
            bias=False,
        )
        self.stride = MERT_CONV_STRIDES[layer_id]
        self.kernel = MERT_CONV_KERNELS[layer_id]

    # kernel/stride vary per instance → not statically trackable
    @override
    def forward[B: IntVar, T: IntVar](
        self, hidden_states: Tensor[[B, 512, T]]
    ) -> Tensor:  # kernel/stride variant per layer_id
        convoluted = F.conv1d(hidden_states, self.conv.weight, bias=None, stride=self.stride)
        return F.gelu(convoluted)


class _FeatureEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [_GroupNormConvLayer()]
        layers += [_PlainConvLayer(i + 1) for i in range(MERT_FEAT_EXTRACT_LAYERS - 1)]
        self.conv_layers = nn.ModuleList(layers)

    @override
    def forward[B: IntVar, T: IntVar, SeqLen: IntVar](self, input_values: Tensor[[B, T]]) -> Tensor[[B, 512, SeqLen]]:
        # heterogeneous loop → not statically trackable; frame count is
        # re-established by the deferred annotation below (runtime-neutral).
        hidden_states: Tensor = input_values[:, None]
        for layer in self.conv_layers:
            hidden_states = layer.forward(hidden_states)
        shaped: Tensor[[B, 512, SeqLen]] = hidden_states  # annotation fallback — boundary receipt above
        return shaped


class MertBackbone(nn.Module):
    """Minimal 7-layer MERT forward mirroring the published trimmed checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _FeatureEncoder()
        self.feature_projection = _FeatureProjection()
        self.encoder = _Encoder()

    @override
    def forward[B: IntVar, T: IntVar, SeqLen: IntVar](
        self, input_values: Tensor[[B, T]]
    ) -> tuple[Tensor[[B, SeqLen, 768]], tuple[Tensor[[B, SeqLen, 768]], ...]]:
        features = self.feature_extractor.forward(input_values)
        projected = self.feature_projection.forward(features.transpose(1, 2))
        last_hidden_state, hidden_states = self.encoder.forward(projected)
        return last_hidden_state, hidden_states


def load_mert_backbone(snapshot: Path, *, device: torch.device, dtype: torch.dtype) -> MertBackbone:
    """Build the backbone from a decompressed asset bundle and move it to ``device``.

    Safetensors keys mirror the attribute tree of :class:`MertBackbone`; the
    positional conv's ``load_state_dict`` translation of the checkpoint's
    legacy ``weight_g``/``weight_v`` keys is handled by torch on the weight-norm
    parametrization. Extra checkpoint keys must be unused-at-eval
    fields (e.g. ``masked_spec_embed``) or loading fails.
    """
    _load_config(snapshot)
    state = load_file(str(snapshot / "model.safetensors"), device="cpu")
    # Unused at eval; upstream initializes it but inference never touches it.
    _ = state.pop("masked_spec_embed", None)

    backbone = MertBackbone()
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if unexpected or missing:
        message = f"MERT checkpoint mismatch (missing: {missing}, unexpected: {unexpected})"
        raise ValueError(message)
    return backbone.to(device=device, dtype=dtype).eval()
