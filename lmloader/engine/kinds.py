"""Engine kinds — registry of runnable model-engine types.

`llama` (PrismML fork) is the built-in engine for GGUF/GGML. Other kinds can
be discovered on disk or added as custom engines; each kind declares the
binaries to look for, the model formats it runs, and the launch-arg template
used to start a server for a model of that format.
"""

ENGINE_KINDS: dict[str, dict] = {
    "llama": {
        "label": "llama.cpp / PrismML 三进制 (GGUF)",
        "binary_names": ["llama-server.exe", "llama-server"],
        "formats": [".gguf", ".ggml"],
        # Special-cased in lifecycle/router — full llama arg set is built there.
        "args_template": None,
    },
    "rwkv": {
        "label": "RWKV (rwkv.cpp)",
        "binary_names": ["rwkv-server.exe", "rwkv_server"],
        "formats": [".bin"],
        "args_template": "--model {model} --port {port} --host {host}",
    },
    "onnx": {
        "label": "ONNX Runtime (onnxruntime-genai)",
        "binary_names": ["onnxruntime-genai.exe", "onnxruntime_genai"],
        "formats": [".onnx"],
        "args_template": "--model {model} --port {port} --host {host}",
    },
    "tensorrt": {
        "label": "TensorRT-LLM",
        "binary_names": ["trtllm-serve", "trtllm-serve.exe"],
        "formats": [".engine"],
        "args_template": "--model {model} --port {port} --host {host}",
    },
    "vllm": {
        "label": "vLLM (HuggingFace / Safetensors)",
        # pip entry point: vllm.exe (Windows) / vllm (Linux) in the env's
        # Scripts directory. Add that directory to the engine scan dirs.
        "binary_names": ["vllm.exe", "vllm"],
        # Only .safetensors / HF model directories. GGUF/GGML are deliberately
        # NOT claimed here so the built-in primary (binary) and ternary engines
        # keep handling every GGUF model.
        "formats": [".safetensors"],
        # Uses a `serve` subcommand with a positional model (dir) path; the arg
        # list is built directly in router (like llama), so no template here.
        "args_template": None,
    },
    "d3": {
        "label": "Decision 3.0 (System One 决策模型)",
        # A d3 package ships its own server script (d3_server.py, standard
        # library HTTP) and is run as `python <pkg>/d3_server.py ...`, so the
        # "binary" is a Python interpreter and the script becomes arg #1. Any
        # venv containing python.exe on the engine scan dirs qualifies.
        "binary_names": ["python.exe", "python"],
        # Directory-signature based (MODEL_MANIFEST.json), not extension based:
        # claiming .safetensors here would fight vLLM over every HF dir.
        "formats": [],
        "args_template": None,
    },
    "d3-gguf": {
        "label": "llama.cpp d3 GGUF 决策引擎",
        # A decision model compiled to GGUF is identified by its
        # `<arch>.decision.type` metadata, not by its extension, so this kind
        # deliberately claims no formats and no scan-dir binaries: it never
        # hijacks an ordinary GGUF model and can only be added as a custom
        # engine. The launch args are the ordinary llama set (built in router).
        "binary_names": [],
        "formats": [],
        "args_template": None,
    },
}


def kind_label(kind: str) -> str:
    k = ENGINE_KINDS.get(kind)
    return k["label"] if k else kind


def formats_for(kind: str) -> list[str]:
    k = ENGINE_KINDS.get(kind)
    return list(k["formats"]) if k else []


def kinds_for_format(ext: str) -> list[str]:
    """Engine kinds that can run a model file with this lowercase extension."""
    ext = ext.lower()
    return [k for k, v in ENGINE_KINDS.items() if ext in v["formats"]]


MODEL_FORMATS = {
    ".gguf": "GGUF",
    ".ggml": "GGML",
    ".bin": "RWKV",
    ".onnx": "ONNX",
    ".engine": "TensorRT",
    ".safetensors": "Safetensors",
}


def format_of(path: str) -> str:
    """Human-readable model format for a file path ('' if unknown)."""
    p = str(path).lower()
    for ext, label in MODEL_FORMATS.items():
        if p.endswith(ext):
            return label
    return ""
