"""Tests for GGUF parser — quantization detection and Bonsai support."""

import pytest
from unittest.mock import MagicMock, patch

from lmloader.infra.gguf_parser import (
    _file_type_to_quant,
    _get_quantization_from_tensors,
    _get_quantization,
    _BONSAI_QUANT_MAP,
    _TERNARY_TENSOR_TYPES,
)


def test_file_type_to_quant_standard():
    assert _file_type_to_quant(12) == "Q4_K"   # gguf>=0.19 ID for Q4_K
    assert _file_type_to_quant(8) == "Q8_0"
    assert _file_type_to_quant(1) == "F16"
    assert _file_type_to_quant(15) == "Q8_K"


def test_file_type_to_quant_bonsai():
    assert _file_type_to_quant(141) == "PQ2_0_TERNARY"
    assert _file_type_to_quant(143) == "TL1_TERNARY"
    assert _file_type_to_quant(144) == "TL2_TERNARY"


def test_file_type_to_quant_unknown():
    result = _file_type_to_quant(250)
    assert result == "CUSTOM_250"


def test_bonsai_quant_map_values():
    assert _BONSAI_QUANT_MAP[141] == "PQ2_0_TERNARY"
    assert _BONSAI_QUANT_MAP[142] == "PQ2_0_TERNARY"
    assert _BONSAI_QUANT_MAP[143] == "TL1_TERNARY"
    assert _BONSAI_QUANT_MAP[144] == "TL2_TERNARY"


def test_tensor_fallback_detects_ternary():
    reader = MagicMock()
    tensor1 = MagicMock()
    tensor1.tensor_type = "F16"
    tensor2 = MagicMock()
    tensor2.tensor_type = "PQ2_0_TERNARY"
    reader.tensors = [tensor1, tensor2]

    result = _get_quantization_from_tensors(reader)
    assert result == "PQ2_0_TERNARY"


def test_tensor_fallback_returns_most_common_for_standard():
    # No special quantization present: the most frequent tensor type is the hint.
    reader = MagicMock()
    tensor = MagicMock()
    tensor.tensor_type = "Q4_K_M"
    reader.tensors = [tensor]

    result = _get_quantization_from_tensors(reader)
    assert result == "Q4_K_M"


def test_tensor_fallback_prefers_nvfp4_over_common():
    # Mixed-quant NVFP4-Q4_K_M files: NVFP4 tensors win over the more numerous
    # F32/Q4_K bookkeeping tensors.
    reader = MagicMock()
    types = ["F32", "F32", "NVFP4", "NVFP4", "Q4_K", "Q4_K"]
    tensors = []
    for t in types:
        tk = MagicMock()
        tk.tensor_type = t
        tensors.append(tk)
    reader.tensors = tensors

    result = _get_quantization_from_tensors(reader)
    assert result == "NVFP4"


def test_tensor_fallback_normalizes_int_ids():
    # gguf's GGUFReader returns tensor_type as an int ID; the scan must map it
    # through the same file_type→name table as general.file_type.
    reader = MagicMock()
    t1 = MagicMock()
    t1.tensor_type = 0      # F32
    t2 = MagicMock()
    t2.tensor_type = 40     # NVFP4
    reader.tensors = [t1, t2]

    result = _get_quantization_from_tensors(reader)
    assert result == "NVFP4"


def test_quant_conflict_uses_tensor_for_ordinary_types():
    # A Q4_K_M file whose general.file_type is the stale approximation Q8_K
    # (converter bug) must report the actual tensor layout, not the tag.
    reader = MagicMock()
    fields = {"general.file_type": MagicMock()}
    fields["general.file_type"].parts = [None, None, None, [15]]   # Q8_K
    fields["general.file_type"].types = [None]
    reader.fields = fields

    t1 = MagicMock(); t1.tensor_type = 12; t1.type = 12   # Q4_K
    t2 = MagicMock(); t2.tensor_type = 0;  t2.type = 0    # F32
    t3 = MagicMock(); t3.tensor_type = 12; t3.type = 12
    t4 = MagicMock(); t4.tensor_type = 12; t4.type = 12
    reader.tensors = [t1, t2, t3, t4]

    result = _get_quantization(reader)
    assert result == "Q4_K"


def test_quant_consistent_file_type_wins():
    # Real Q8_K files: tensor census agrees with general.file_type.
    reader = MagicMock()
    fields = {"general.file_type": MagicMock()}
    fields["general.file_type"].parts = [None, None, None, [15]]   # Q8_K
    fields["general.file_type"].types = [None]
    reader.fields = fields

    t1 = MagicMock(); t1.tensor_type = 15; t1.type = 15   # Q8_K
    t2 = MagicMock(); t2.tensor_type = 0;  t2.type = 0    # F32
    reader.tensors = [t1, t2]

    result = _get_quantization(reader)
    assert result == "Q8_K"


def test_tensor_fallback_handles_exception():
    reader = MagicMock()
    tensor = MagicMock()
    type(tensor).tensor_type = property(lambda self: (_ for _ in ()).throw(ValueError("unknown type")))
    type(tensor).type = property(lambda self: 999)
    reader.tensors = [tensor]

    result = _get_quantization_from_tensors(reader)
    assert result is None or isinstance(result, str)
