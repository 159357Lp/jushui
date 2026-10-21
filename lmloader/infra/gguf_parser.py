"""GGUF file parser for extracting model metadata."""

import re
import struct
from pathlib import Path
from typing import Optional

import numpy as np

import gguf
from gguf.gguf_reader import GGUFValueType

from lmloader.logger import logger

# Bump whenever parse_gguf gains a field an existing record cannot compute from
# what it already stores. scanner._fingerprint mixes this into its cache key, so
# a new field reaches already-registered models on the next scan instead of
# silently staying absent (the planner would then fall back forever).
PARSER_SCHEMA = 4

_BONSAI_QUANT_MAP = {
    141: "PQ2_0_TERNARY",
    142: "PQ2_0_TERNARY",
    143: "TL1_TERNARY",
    144: "TL2_TERNARY",
}

# Approximate block/type sizes for Bonsai ternary types. Used only for tensor
# offset math during metadata parsing (tensor data is never read), so exact
# values are not required here.
_BONSAI_QUANT_SIZES = {
    141: (256, 32),
    142: (256, 32),
    143: (256, 64),
    144: (256, 64),
}

_TERNARY_TENSOR_TYPES = {
    "PQ2_0_TERNARY", "TL1_TERNARY", "TL2_TERNARY",
    "TQ1_0", "TQ2_0", "I2_S", "TL1", "TL2",
}

# Tensor-level "special" quantizations that general.file_type reports only
# approximately (mixed-quant files such as NVFP4-Q4_K_M often carry
# file_type=Q8_K while the bulk of the weights are NVFP4/MXFP4). When any of
# these is found on a tensor, it wins over the file_type label.
_SPECIAL_TENSOR_TYPES = _TERNARY_TENSOR_TYPES | {"MXFP4", "NVFP4"}


def _register_bonsai_types():
    """Extend GGMLQuantizationType and GGML_QUANT_SIZES with Bonsai ternary IDs.

    gguf<=0.19 does not know IDs 141-144; GGUFReader raises
    "np.uint32(142) is not a valid GGMLQuantizationType" when parsing the tensor
    list, which breaks parsing of Bonsai models entirely. Runs once at import.
    """
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

    for value, name in _BONSAI_QUANT_MAP.items():
        if value in GGMLQuantizationType._value2member_map_:
            continue
        # GGMLQuantizationType is an IntEnum: create the member via int.__new__
        member = int.__new__(GGMLQuantizationType, value)
        member._name_ = name
        member._value_ = value
        GGMLQuantizationType._value2member_map_[value] = member
        GGMLQuantizationType._member_map_[name] = member
        if name not in GGMLQuantizationType._member_names_:
            GGMLQuantizationType._member_names_.append(name)
        GGML_QUANT_SIZES.setdefault(member, _BONSAI_QUANT_SIZES[value])


_register_bonsai_types()

# Routed-expert tensor-name markers, and the shared-expert names that must not
# be mistaken for them. See _read_expert_shape.
_ROUTED_EXPERT_MARKERS = ("_exps", "expw")
_SHARED_EXPERT_MARKERS = ("shexp", "shared_exp")


class GGUFMetadata:
    def __init__(
        self,
        architecture: str,
        quantization: str,
        parameters: str,
        tensor_count: int,
        kv_count: int,
        context_length: int,
        embedding_length: int,
        feed_forward_length: int,
        attention_head_count: int,
        block_count: int,
        supports_mtp: bool,
        raw_metadata: dict,
        decision_type: str = "",
        vocab_size: int = 0,
        expert_count: int = 0,
        expert_used: int = 0,
        moe_layer_count: int = 0,
        expert_bytes: int = 0,
        layer_bytes: int = 0,
        tail_bytes: int = 0,
        blk_groups: int = 0,
        attention_head_count_kv: int = 0,
        attention_key_length: int = 0,
        attention_value_length: int = 0,
        full_attention_interval: int = 0,
        ssm_conv_kernel: int = 0,
        ssm_state_size: int = 0,
        ssm_group_count: int = 0,
        ssm_inner_size: int = 0,
    ):
        self.architecture = architecture
        self.quantization = quantization
        self.parameters = parameters
        self.tensor_count = tensor_count
        self.kv_count = kv_count
        self.context_length = context_length
        self.embedding_length = embedding_length
        self.feed_forward_length = feed_forward_length
        self.attention_head_count = attention_head_count
        self.block_count = block_count
        self.supports_mtp = supports_mtp
        self.decision_type = decision_type
        self.vocab_size = vocab_size
        self.expert_count = expert_count
        self.expert_used = expert_used
        self.moe_layer_count = moe_layer_count
        self.expert_bytes = expert_bytes
        self.layer_bytes = layer_bytes
        self.tail_bytes = tail_bytes
        self.blk_groups = blk_groups
        self.attention_head_count_kv = attention_head_count_kv
        self.attention_key_length = attention_key_length
        self.attention_value_length = attention_value_length
        self.full_attention_interval = full_attention_interval
        self.ssm_conv_kernel = ssm_conv_kernel
        self.ssm_state_size = ssm_state_size
        self.ssm_group_count = ssm_group_count
        self.ssm_inner_size = ssm_inner_size
        self.raw_metadata = raw_metadata

    @property
    def is_moe(self) -> bool:
        # Both halves must be present: a converter that omits expert_count but
        # still stores `.exps.` tensors (or the reverse) leaves the offload
        # planner without a per-expert byte cost, so it must fall back to dense
        # handling rather than guess.
        return self.expert_count > 0 and self.expert_bytes > 0

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "quantization": self.quantization,
            "parameters": self.parameters,
            "tensor_count": self.tensor_count,
            "kv_count": self.kv_count,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "feed_forward_length": self.feed_forward_length,
            "attention_head_count": self.attention_head_count,
            "block_count": self.block_count,
            "supports_mtp": self.supports_mtp,
            "decision_type": self.decision_type,
            "vocab_size": self.vocab_size,
            "expert_count": self.expert_count,
            "expert_used": self.expert_used,
            "moe_layer_count": self.moe_layer_count,
            "expert_bytes": self.expert_bytes,
            "layer_bytes": self.layer_bytes,
            "tail_bytes": self.tail_bytes,
            "blk_groups": self.blk_groups,
            "attention_head_count_kv": self.attention_head_count_kv,
            "attention_key_length": self.attention_key_length,
            "attention_value_length": self.attention_value_length,
            "full_attention_interval": self.full_attention_interval,
            "ssm_conv_kernel": self.ssm_conv_kernel,
            "ssm_state_size": self.ssm_state_size,
            "ssm_group_count": self.ssm_group_count,
            "ssm_inner_size": self.ssm_inner_size,
        }


_KV_FIXED_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                  10: 8, 11: 8, 12: 8}


def _skip_gguf_value(fh, vtype: int):
    """按 GGUF 类型跳过 KV 值。只读长度，不解释内容。"""
    if vtype == 8:                      # string
        fh.read(struct.unpack("<Q", fh.read(8))[0])
    elif vtype == 9:                    # array: 元素类型 + 个数
        et = struct.unpack("<I", fh.read(4))[0]
        n = struct.unpack("<Q", fh.read(8))[0]
        if et in _KV_FIXED_SIZE:
            fh.read(_KV_FIXED_SIZE[et] * n)
        else:
            for _ in range(n):          # 变长元素（字符串/嵌套数组）逐个走
                _skip_gguf_value(fh, et)
    elif vtype in _KV_FIXED_SIZE:
        fh.read(_KV_FIXED_SIZE[vtype])
    else:
        raise ValueError(f"未知 GGUF KV 类型 {vtype}")


def _walk_tensor_sizes(path: Path) -> tuple[int, int, int]:
    """返回 (blk.* 张量字节合计, 其余重尾字节合计, blk 层组数)。

    只走张量表，用相邻 offset 相减得到每个张量的真实字节：既不需要 mmap
    （20GB 级文件在 Windows 上会 WinError 1455 直接失败），也不依赖 ggml 类型
    表认不认识 NVFP4/PQ2_0_TERNARY 这些自定义量化——所以量化模型的重尾大小
    是文件里读出来的，而不是按 f16 猜的。
    """
    layer_bytes = tail_bytes = 0
    groups = set()
    with open(path, "rb") as fh:
        head = fh.read(24)
        if len(head) < 24 or head[:4] != b"GGUF":
            raise ValueError("不是 GGUF 文件头")
        n_tensors = struct.unpack_from("<Q", head, 8)[0]
        n_kv = struct.unpack_from("<Q", head, 16)[0]
        if not 0 < n_tensors < 1_000_000 or n_kv > 100_000:
            raise ValueError(f"张量/KV 数量不合理：{n_tensors}/{n_kv}")
        align = 32
        for _ in range(n_kv):
            klen = struct.unpack("<Q", fh.read(8))[0]
            key = fh.read(klen)
            vtype = struct.unpack("<I", fh.read(4))[0]
            if key == b"general.alignment" and vtype in (2, 3, 4, 5):
                raw = fh.read(_KV_FIXED_SIZE[vtype])
                align = abs(int.from_bytes(raw, "little",
                                           signed=vtype in (3, 5))) or 32
                continue
            _skip_gguf_value(fh, vtype)
        tensors = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", fh.read(8))[0]
            name = fh.read(nlen).decode("utf-8", "replace")
            nd = struct.unpack("<I", fh.read(4))[0]
            fh.read(8 * nd)                                  # dims
            fh.read(4)                                       # dtype
            tensors.append((name, struct.unpack("<Q", fh.read(8))[0]))
        pos = fh.tell()
        data_start = (pos + align - 1) // align * align
        total = path.stat().st_size
        tensors.sort(key=lambda t: t[1])
        for i, (name, off) in enumerate(tensors):
            nxt = tensors[i + 1][1] if i + 1 < len(tensors) else total - data_start
            size = max(0, nxt - off)
            if name.startswith("blk."):
                layer_bytes += size
                head = name.split(".", 2)
                if len(head) > 1 and head[1].isdigit():
                    groups.add(int(head[1]))
            else:
                tail_bytes += size
    if layer_bytes + tail_bytes > total:
        raise ValueError("张量字节合计超过文件大小：offset 推算法不成立")
    return layer_bytes, tail_bytes, len(groups)


def parse_gguf(file_path: str, force: bool = False) -> Optional[GGUFMetadata]:
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    # Ollama stores blobs as extension-less files (sha256-<digest>); their
    # content is still GGUF. force=True parses by content/magic instead of
    # relying on the .gguf suffix (GGUFReader validates the magic itself).
    if not force and path.suffix.lower() != ".gguf":
        logger.error(f"Not a GGUF file: {file_path}")
        return None

    try:
        reader = gguf.GGUFReader(str(path))

        architecture = _get_architecture(reader)
        quantization = _get_quantization(reader)
        parameters = _get_parameters(reader, architecture)
        tensor_count = len(reader.tensors)
        kv_count = len(reader.fields)

        ctx_key = f"{architecture}.context_length"
        emb_key = f"{architecture}.embedding_length"
        ff_key = f"{architecture}.feed_forward_length"
        head_key = f"{architecture}.attention.head_count"
        block_key = f"{architecture}.block_count"

        context_length = _get_int(reader, ctx_key) or _get_int(reader, "llama.context_length") or 0
        embedding_length = _get_int(reader, emb_key) or _get_int(reader, "llama.embedding_length") or 0
        feed_forward_length = _get_int(reader, ff_key) or _get_int(reader, "llama.feed_forward_length") or 0
        attention_head_count = _get_int(reader, head_key) or _get_int(reader, "llama.attention.head_count") or 0
        block_count = _get_int(reader, block_key) or _get_int(reader, "llama.block_count") or 0

        # MTP (multi-token prediction) draft layers are embedded in the GGUF as
        # extra tensors. Naming varies by converter: "mtp.layers..." (DeepSeek /
        # official Qwen), "blk.N.nextn.*" (llama.cpp Qwen3-Next/3.5) or
        # "...draft...". Tensor-content detection is reliable; filename hints
        # remain only as a fallback in the engine router.
        mtp_markers = ("mtp", "nextn", "draft")
        supports_mtp = any(
            any(m in str(getattr(t, "name", "")).lower() for m in mtp_markers)
            for t in reader.tensors
        )

        raw_metadata = {}
        for field_name, field in reader.fields.items():
            if field_name.startswith("GGUF."):
                continue
            val = _read_field_value(field)
            if val is not None and isinstance(val, str) and len(val) < 1000:
                raw_metadata[field_name] = val

        # Vocab size feeds the offload planner's heavy-tail reserve (input
        # embedding + output table dwarf an average trunk layer on 150k+-vocab
        # models). It is an ARRAY field, so the element count is the value.
        vocab_size = 0
        tok_field = reader.fields.get("tokenizer.ggml.tokens")
        if tok_field is not None:
            try:
                n = len(tok_field.parts)
                if 0 < n < 10_000_000:
                    vocab_size = n
            except (TypeError, AttributeError):
                pass

        metadata = GGUFMetadata(
            architecture=architecture,
            quantization=quantization,
            parameters=parameters,
            tensor_count=tensor_count,
            kv_count=kv_count,
            context_length=context_length,
            embedding_length=embedding_length,
            feed_forward_length=feed_forward_length,
            attention_head_count=attention_head_count,
            block_count=block_count,
            supports_mtp=supports_mtp,
            raw_metadata=raw_metadata,
            decision_type=_get_string(reader, f"{architecture}.decision.type")
            or next((v for k, v in raw_metadata.items()
                     if k.endswith(".decision.type")), ""),
            vocab_size=vocab_size,
            **_read_expert_shape(reader, architecture),
            **_read_kv_shape(reader, architecture, embedding_length,
                             attention_head_count),
            **_read_ssm_shape(reader, architecture),
        )

        # 实测张量字节：分层解算器靠这两个数区分『块层』与『重尾』，比按
        # vocab×hidden×2 猜 f16 准得多（NVFP4/三元量化的表远小于 f16 猜测）。
        # 只有 blk 组数与层算术对得上才发布，对不上就留 0 让 planner 退回估算
        # ——宁可退回粗估，也不能把对不上号的字节当成每层均值拿去解 -ngl。
        # MTP 草稿的落法有两种：独立 blk.N 组（Qwen3.8-Coder 那种）或嵌在最后一
        # 组里（NVFP4-LOW 实测 blk.64.nextn.*，65 组而 block_count=65）。旧判据
        # 只认前者，把这只 flagship 的实测字节整个拦掉，planner 至今仍按 f16 猜
        # 重尾——两种都对上号才发布，组数照实发布给 planner 当层数。
        try:
            lb, tb, groups = _walk_tensor_sizes(path)
            allowed = {block_count} | ({block_count + 1} if supports_mtp else set())
            if groups and groups in allowed:
                metadata.layer_bytes, metadata.tail_bytes = lb, tb
                metadata.blk_groups = groups
            else:
                logger.warning(
                    f"Tensor walk skipped for {path.name}: {groups} blk groups "
                    f"not in {sorted(allowed)} (block_count={block_count}, "
                    f"mtp={int(supports_mtp)})")
        except Exception as e:
            logger.warning(f"Tensor size walk failed for {path.name}: {e}")

        logger.info(f"Parsed GGUF: {path.name} - {architecture} {quantization} {parameters}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to parse GGUF {file_path}: {e}")
        return None


def _read_field_value(field) -> Optional[object]:
    if not hasattr(field, 'parts') or not field.parts:
        return None
    if not hasattr(field, 'types') or not field.types:
        return None

    vtype = field.types[0]

    if vtype == GGUFValueType.STRING:
        if len(field.parts) >= 5:
            try:
                return bytes(field.parts[4]).decode("utf-8", errors="replace")
            except Exception:
                return None
        return None

    if vtype in (GGUFValueType.UINT32, GGUFValueType.INT32):
        if len(field.parts) >= 4:
            try:
                return int(field.parts[3][0])
            except Exception:
                return None
        return None

    if vtype in (GGUFValueType.UINT64, GGUFValueType.INT64):
        if len(field.parts) >= 4:
            try:
                return int(field.parts[3][0])
            except Exception:
                return None
        return None

    if vtype == GGUFValueType.FLOAT32:
        if len(field.parts) >= 4:
            try:
                return float(field.parts[3][0])
            except Exception:
                return None
        return None

    if vtype == GGUFValueType.BOOL:
        if len(field.parts) >= 4:
            try:
                return bool(field.parts[3][0])
            except Exception:
                return None
        return None

    return None


def _get_architecture(reader: gguf.GGUFReader) -> str:
    arch = _get_string(reader, "general.architecture")
    return arch if arch else "unknown"


def _get_quantization(reader: gguf.GGUFReader) -> str:
    file_type = _get_int(reader, "general.file_type")

    tensor_quant = _get_quantization_from_tensors(reader)  # FIXED: B4

    if file_type is not None:
        result = _file_type_to_quant(file_type)
        if tensor_quant and tensor_quant != result:
            # general.file_type is written by the converter and frequently
            # stale/approximate (Q4_K_M files are routinely tagged Q8_K, and
            # mixed-quant files only store their dominant type). The tensor
            # census is hard evidence of the actual stored layout, so it wins
            # for both special (NVFP4/MXFP4/ternary) and ordinary conflicts.
            logger.warning(f"general.file_type={result} conflicts with tensor type={tensor_quant}, using tensor type")
            return tensor_quant
        return result

    if tensor_quant:
        logger.info(f"general.file_type missing, using tensor type fallback: {tensor_quant}")
        return tensor_quant

    quant_ver = _get_int(reader, "general.quantization_version")
    if quant_ver is not None:
        return f"v{quant_ver}"

    return "unknown"


def _get_quantization_from_tensors(reader: gguf.GGUFReader) -> Optional[str]:
    """Scan tensors for the dominant special quantization type.

    Used as the authority for mixed-quant files (e.g. NVFP4-Q4_K_M) where
    ``general.file_type`` only stores an approximation like Q8_K, and as a
    fallback when that key is missing entirely. The most frequent special
    type wins (NVFP4 > MXFP4 > ternary), then plain Q8_K/Q4_K as a fallback.
    """
    counts: dict[str, int] = {}
    for tensor in reader.tensors:
        try:
            type_name = tensor.tensor_type if hasattr(tensor, 'tensor_type') else None
            # gguf's GGUFReader yields tensor_type as an int ID (GGML
            # quantization enum), not a string — normalize via the shared map.
            if isinstance(type_name, int):
                type_name = _file_type_to_quant(type_name)
            if type_name is None:
                type_val = int(tensor.type) if hasattr(tensor, 'type') else None
                if type_val is not None:
                    type_name = _file_type_to_quant(type_val)
            if type_name:
                counts[type_name] = counts.get(type_name, 0) + 1
        except Exception:
            continue
    if not counts:
        return None
    for special in ("NVFP4", "MXFP4", "PQ2_0_TERNARY", "TL1_TERNARY",
                    "TL2_TERNARY", "TQ1_0", "TQ2_0", "I2_S", "TL1", "TL2"):
        if counts.get(special):
            return special
    # No special type: use the most frequent tensor type as a hint.
    return max(counts, key=counts.get)


def _get_parameters(reader: gguf.GGUFReader, architecture: str = "") -> str:
    param_count = _get_int(reader, "general.parameter_count")
    if param_count:
        return _format_parameters(param_count)

    size_label = _get_string(reader, "general.size_label")
    if size_label:
        return size_label

    emb_key = f"{architecture}.embedding_length" if architecture else "llama.embedding_length"
    block_key = f"{architecture}.block_count" if architecture else "llama.block_count"
    embedding_length = _get_int(reader, emb_key) or 0
    block_count = _get_int(reader, block_key) or 0
    if embedding_length and block_count:
        estimated = embedding_length * block_count * 4
        return _format_parameters(estimated)

    return "unknown"


def _read_int_alias(reader, architecture, *suffixes):
    """Read the first `{arch}.{suffix}` (or any `*.{suffix}`) int field that exists.

    Converters disagree twice over: on the arch prefix (a qwen3moe file may
    store its expert fields under a name that does not match
    general.architecture) and on the field name itself (qwen3moe ships
    `expert_used_count`, llama.cpp's own writer says `expert_used`, deepseek2
    says `n_routed_experts`). An exact-key miss is therefore not evidence that
    the model lacks the value, and neither is a suffix miss.
    """
    for suffix in suffixes:
        value = _get_int(reader, f"{architecture}.{suffix}") or 0
        if value:
            return value
        for key in reader.fields:
            if key.startswith("GGUF.") or not key.endswith(f".{suffix}"):
                continue
            value = _get_int(reader, key) or 0
            if value:
                return value
    return 0


def _read_expert_shape(reader: gguf.GGUFReader, architecture: str) -> dict:
    """Collect the routed-expert geometry the offload planner needs.

    Only *routed* experts count: they are the tensors `-ot`/`-ncmoe` can keep
    on CPU, while shared experts (`shexp`) fire on every token and must stay
    resident. Naming is converter-dependent (`_exps` for qwen3moe/deepseek2,
    `expw` for gpt-oss), so the census is by tensor name rather than one
    hardcoded pattern, and a layer count read from metadata wins over the
    number of layers that happened to carry expert tensors.
    """
    expert_count = _read_int_alias(reader, architecture,
                                   "expert_count", "n_routed_experts")
    expert_used = _read_int_alias(reader, architecture,
                                  "expert_used", "expert_used_count",
                                  "num_experts_per_tok", "experts_per_tok")

    expert_bytes = 0
    expert_layers = set()
    packed_experts = 0
    for tensor in reader.tensors:
        name = str(getattr(tensor, "name", ""))
        lowered = name.lower()
        if not any(marker in lowered for marker in _ROUTED_EXPERT_MARKERS):
            continue
        if any(marker in lowered for marker in _SHARED_EXPERT_MARKERS):
            continue
        try:
            expert_bytes += int(tensor.n_bytes)
        except (AttributeError, TypeError, ValueError):
            continue
        layer = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
        if layer:
            expert_layers.add(int(layer.group(1)))
        # A 3-D `_exps` tensor is one whole layer's experts stored together
        # (llama.cpp's layout: expert count is the outermost axis). That is the
        # number the planner divides expert_bytes by, and converters that omit
        # the metadata field still cannot hide it here.
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 3:
            packed_experts = max(packed_experts, int(shape[0]))

    if not expert_count:
        expert_count = packed_experts

    moe_layer_count = _read_int_alias(reader, architecture, "moe_layer_count")
    if not moe_layer_count:
        moe_layer_count = len(expert_layers)

    return {
        "expert_count": expert_count,
        "expert_used": expert_used,
        "moe_layer_count": moe_layer_count,
        "expert_bytes": expert_bytes,
    }


def _read_kv_shape(reader, architecture, embedding_length, head_count) -> dict:
    """Collect the per-token KV-cache geometry the memory budget needs.

    The budget used to assume 2*embedding_length elements per layer per token,
    i.e. one KV set as wide as the whole hidden vector on every layer. Real
    files are narrower on two axes at once: GQA stores `head_count_kv` far below
    `head_count`, and hybrid models (qwen35moe, qwen3next) keep a KV cache only
    on every `full_attention_interval`-th layer — the others are recurrent
    states that cost O(1) per slot, not O(ctx). On the user's 35B-A3B the old
    assumption was 16x too pessimistic: the planner reserved most of a 16GB card
    for a cache llama.cpp itself measured at 128 MiB, which is what starved the
    expert-offload decision it is supposed to feed.
    """
    kv_heads = _read_int_alias(reader, architecture, "attention.head_count_kv")
    k_len = _read_int_alias(reader, architecture, "attention.key_length")
    v_len = _read_int_alias(reader, architecture, "attention.value_length") or k_len
    interval = _read_int_alias(reader, architecture, "full_attention_interval")
    if not kv_heads:
        kv_heads = head_count
    if not (k_len or v_len):
        head_dim = embedding_length // head_count if head_count else 0
        k_len = v_len = head_dim
    return {
        "attention_head_count_kv": kv_heads,
        "attention_key_length": k_len,
        "attention_value_length": v_len,
        "full_attention_interval": interval,
    }


def _read_ssm_shape(reader, architecture) -> dict:
    """Collect the recurrent-state geometry a hybrid model's memory costs.

    A hybrid file (qwen35 / qwen3next / mamba) keeps no KV cache on most of its
    layers — those layers hold a fixed-size recurrent state per sequence slot
    instead. Nothing in the KV budget sees it: the cost does not scale with ctx,
    so a model can ask llama.cpp for a few hundred MiB of state that no existing
    term accounts for, and on the user's 27B NVFP4 flagship the speculative
    draft multiplier makes that 448.88 MiB (measured; see
    offload_planner.recurrent_state_mb for the arithmetic and the anchor).

    Only the llama.cpp-owned field names are accepted. A converter that stores
    the same shapes under other keys yields zeros, and the planner then declines
    to price the state rather than guess it from `embedding_length`.
    """
    return {
        "ssm_conv_kernel": _read_int_alias(reader, architecture,
                                           "ssm.conv_kernel", "mamba.conv_kernel"),
        "ssm_state_size": _read_int_alias(reader, architecture,
                                          "ssm.state_size", "mamba.state_size"),
        "ssm_group_count": _read_int_alias(reader, architecture,
                                           "ssm.group_count", "mamba.n_groups"),
        "ssm_inner_size": _read_int_alias(reader, architecture,
                                          "ssm.inner_size", "mamba.inner_size"),
    }


def _get_string(reader: gguf.GGUFReader, key: str) -> Optional[str]:
    field = reader.fields.get(key)
    if not field:
        return None
    val = _read_field_value(field)
    return val if isinstance(val, str) else None


def _get_int(reader: gguf.GGUFReader, key: str) -> Optional[int]:
    field = reader.fields.get(key)
    if not field:
        return None
    val = _read_field_value(field)
    return int(val) if isinstance(val, (int, float)) else None


def _file_type_to_quant(file_type: int) -> str:
    # Aligned with gguf>=0.19 GGMLQuantizationType IDs (llama.cpp enum).
    quant_map = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K",
        12: "Q4_K",
        13: "Q5_K",
        14: "Q6_K",
        15: "Q8_K",
        16: "IQ2_XXS",
        17: "IQ2_XS",
        18: "IQ3_XXS",
        19: "IQ1_S",
        20: "IQ4_NL",
        21: "IQ3_S",
        22: "IQ2_S",
        23: "IQ4_XS",
        24: "I8",
        25: "I16",
        26: "I32",
        27: "I64",
        28: "F64",
        29: "IQ1_M",
        30: "BF16",
        34: "TQ1_0",
        35: "TQ2_0",
        39: "MXFP4",
        40: "NVFP4",
        41: "Q1_0",
    }
    result = quant_map.get(file_type)
    if result:
        return result
    bonsai = _BONSAI_QUANT_MAP.get(file_type)
    if bonsai:
        return bonsai
    return f"CUSTOM_{file_type}"


def _format_parameters(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)
