"""GGUF file parser for extracting model metadata."""

import struct
from pathlib import Path
from typing import Optional

import numpy as np

import gguf
from gguf.gguf_reader import GGUFValueType

from lmloader.logger import logger

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
        self.raw_metadata = raw_metadata

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
        }


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
        )

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
