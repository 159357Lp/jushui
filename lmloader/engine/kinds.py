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
