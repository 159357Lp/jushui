"""Offline check of gguf_parser._read_expert_shape against a real MoE GGUF.

Reads only the GGUF header + tensor-info table, then feeds those same numbers
to the production function through a stub reader shaped like a gguf-py one.
Works on a partially downloaded file. Run with --break to prove it can go red.
"""
import mmap
import struct
import sys
from collections import Counter

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType

PATH = r"F:/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?',
       10: 'Q', 11: 'q', 12: 'd'}
VT = {0: GGUFValueType.UINT32, 1: GGUFValueType.INT32, 2: GGUFValueType.FLOAT32,
      10: GGUFValueType.UINT64, 11: GGUFValueType.INT64, 12: GGUFValueType.FLOAT64}


def read_header(path):
    with open(path, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:4] != b'GGUF':
            raise SystemExit(f"not GGUF: {mm[:4]!r}")
        off = 4
        _ver, n_tensors, n_kv = struct.unpack_from('<IQq', mm, off)
        off += 20

        def rstr(o):
            ln = struct.unpack_from('<Q', mm, o)[0]
            o += 8
            return mm[o:o + ln].decode('utf-8', 'replace'), o + ln

        def rval(o, t):
            if t == 8:
                return rstr(o)
            if t == 9:
                et, c = struct.unpack_from('<IQ', mm, o)
                o += 12
                vals = []
                for _ in range(c):
                    v, o = rval(o, et)
                    vals.append(v)
                return vals, o
            return struct.unpack_from('<' + FMT[t], mm, o)[0], o + SZ[t]

        fields = {}
        for _ in range(n_kv):
            key, off = rstr(off)
            t, off = struct.unpack_from('<I', mm, off)[0], off + 4
            v, off = rval(off, t)
            fields[key] = (t, v)
        tensors = []
        for _ in range(n_tensors):
            name, off = rstr(off)
            nd, off = struct.unpack_from('<I', mm, off)[0], off + 4
            dims = list(struct.unpack_from('<' + 'Q' * nd, mm, off))
            off += 8 * nd
            ttype, off = struct.unpack_from('<I', mm, off)[0], off + 4
            data_off, off = struct.unpack_from('<Q', mm, off)[0], off + 8
            tensors.append((name, dims, ttype, data_off))
        tail = off
        mm.close()
        return fields, tensors, tail


def n_bytes(dims, ttype):
    elems = int(np.prod(dims)) if dims else 0
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType(ttype)]
    return elems * size // block


class StubField:
    """Shaped so gguf_parser._read_field_value finds the value at parts[3][0]."""

    def __init__(self, value):
        self.types = [VT.get(0, GGUFValueType.UINT32)]
        self.parts = [b'', b'', b'', np.array([value]), None]


class StubTensor:
    def __init__(self, name, dims, ttype):
        self.name = name
        self.shape = np.array(list(reversed(dims)))
        self.tensor_type = GGMLQuantizationType(ttype)
        self.n_bytes = n_bytes(dims, ttype)


class StubReader:
    def __init__(self, fields, tensors):
        self.fields = {k: StubField(v[1]) for k, v in fields.items() if v[0] != 8}
        self.tensors = [StubTensor(n, d, t) for n, d, t, _ in tensors]


def main():
    brk = "--break" in sys.argv
    sys.path.insert(0, r"D:\掬水\lmloader")
    from lmloader.infra import gguf_parser as gp

    fields, tensors, tail = read_header(PATH)
    arch = fields["general.architecture"][1]
    got = gp._read_expert_shape(StubReader(fields, tensors), arch)
    if brk:
        poisoned = StubReader(fields, tensors)
        for t in poisoned.tensors:
            if "_exps" in t.name:
                t.n_bytes = 0
                break
        got = gp._read_expert_shape(poisoned, arch)

    routed = [t for t in tensors if "_exps" in t[0] or "expw" in t[0]]
    shared = [t for t in tensors if "shexp" in t[0] or "shared_exp" in t[0]]
    total = sum(n_bytes(d, t) for _n, d, t, _o in tensors)
    print(f"architecture          {arch}")
    print(f"tensors               {len(tensors)}   header table ends at {tail}")
    print(f"all-tensor bytes      {total}  ({total / 1073741824:.2f} GiB)")
    print(f"routed-expert tensors {len(routed)}    shared-expert tensors {len(shared)}")
    print(f"expert types          {Counter(t for _n, _d, t, _o in routed).most_common()}")
    print(f"one routed tensor     {routed[0][0]} shape={routed[0][1]} -> {n_bytes(routed[0][1], routed[0][2]) / 1048576:.1f} MiB")
    print(f"--- parsed by production code ---")
    print(f"expert_count          {got['expert_count']}   (header: {fields.get(f'{arch}.expert_count', ('', 0))[1]})")
    print(f"expert_used           {got['expert_used']}   (header candidates: "
          + ", ".join(f"{k}={v[1]}" for k, v in fields.items() if 'used' in k) + ")")
    print(f"moe_layer_count       {got['moe_layer_count']}   (block_count: {fields.get(f'{arch}.block_count', ('', 0))[1]})")
    print(f"expert_bytes          {got['expert_bytes']}  ({got['expert_bytes'] / 1073741824:.2f} GiB)")

    problems = []
    hc = fields.get(f"{arch}.expert_count", ("", None))[1]
    if hc is not None and got["expert_count"] != hc:
        problems.append(f"expert_count {got['expert_count']} != header {hc}")
    hu = next((v[1] for k, v in fields.items() if k.endswith("_expert_used") or k.endswith("expert_used_count")), None)
    if hu is not None and got["expert_used"] != hu:
        problems.append(f"expert_used {got['expert_used']} != header {hu}")
    if got["moe_layer_count"] != fields.get(f"{arch}.block_count", ("", -1))[1]:
        problems.append("moe_layer_count != block_count")
    if shared:
        problems.append(f"{len(shared)} shared-expert tensors present; check they were excluded")
    expect = sum(n_bytes(d, t) for n, d, t, _o in routed)
    if got["expert_bytes"] != expect:
        problems.append(f"expert_bytes {got['expert_bytes']} != routed sum {expect}")
    if problems:
        print("\nRED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nGREEN: parser agrees with the header on every expert number")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
