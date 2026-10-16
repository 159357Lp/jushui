"""Verification: vLLM kind registered + custom-engine endpoint works when the
guardian is approved. Run with the project's Python 3.12.
"""
from pathlib import Path

from fastapi.testclient import TestClient

import lmloader.config as config
from lmloader.api.app import app
from lmloader.infra import guardian as g_mod
from lmloader.infra.engine_store import engine_dir

config.settings._config.server.api_token = ""


def main():
    # Auto-approve every guardian request (what the GUI does on click).
    # Assigned on the instance, so no self binding.
    async def _always_approve(req, timeout=300):
        return True
    g_mod.config_guardian.wait_for_approval = _always_approve

    with TestClient(app) as client:
        # 1) vllm kind is declared
        kinds = client.get("/api/engine/kinds").json()["kinds"]
        vllm = next((k for k in kinds if k["kind"] == "vllm"), None)
        assert vllm is not None, "vllm kind missing"
        print("vllm kind:", vllm["label"], "| formats:", vllm["formats"],
              "| binaries:", vllm["binary_names"])
        assert ".gguf" not in vllm["formats"] and ".ggml" not in vllm["formats"], \
            "vllm must NOT claim gguf/ggml (would affect binary/ternary)"
        print("OK: vllm does not claim .gguf/.ggml")

        # 2) the failing endpoint succeeds once approval is granted
        binary = engine_dir().resolve() / "llama-server.exe"
        if binary.is_file():
            resp = client.post("/api/engine/custom", json={
                "name": "probe-verify", "kind": "llama", "binary_path": str(binary)})
            print("custom endpoint status with approval:", resp.status_code)
            assert resp.status_code == 200, resp.text
            # clean up the probe record
            import yaml
            cfg_p = Path("config.yaml")
            data = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
            remaining = [c for c in data.get("custom_engines", [])
                         if c.get("name") != "probe-verify"]
            if remaining:
                data["custom_engines"] = remaining
            else:
                data.pop("custom_engines", None)
            cfg_p.write_text(yaml.safe_dump(data, allow_unicode=True,
                                            sort_keys=False), encoding="utf-8")
        else:
            print("(skip custom-endpoint check: no bundled binary)")

    # 3) vLLM launch config for a HuggingFace directory
    import sys
    from unittest.mock import MagicMock
    er_mod = sys.modules["lmloader.engine.router"]
    from lmloader.engine.router import router as er

    # Simulate a discovered vLLM engine (no real vllm installed in this env).
    er_mod.find_engine_by_kind = lambda kind, discovered=None: {
        "name": "vllm@vllm", "kind": "vllm",
        "binary_path": "D:/engines/vllm.exe",
        "formats": [".safetensors"], "source": "scan",
    }

    m = MagicMock()
    m.name = "hf-model"
    m.path = "D:/models/hf-model"
    m.metadata = {"hf_model": True}
    m.backend = None
    lc = er.build_launch_config(m, 8123)
    print("vllm launch args:", lc.args)
    assert lc.args == ["serve", "D:/models/hf-model", "--host", "127.0.0.1",
                       "--port", "8123"]
    print("ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
