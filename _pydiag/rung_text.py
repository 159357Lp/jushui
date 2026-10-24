"""一档一测：写配置 -> 上线 -> 关视觉 -> 跑四条固定负载 -> 从运行时表读数。

只走应用自己的接口，不碰引擎命令行；速度、占用、加载前后空闲量全部来自
GET /api/models/run-stats，也就是速度档案页看到的那一行。
"""
import sys, time, json
sys.path.insert(0, "_pydiag")
import calib_loop as cl

ngl = int(sys.argv[1]); ctx = int(sys.argv[2])
print("cfg", json.dumps(cl.set_cfg(ngl, ctx), ensure_ascii=False)[:90])
cl.api("/api/models/unload", method="POST", query={"model_name": cl.MODEL}); time.sleep(2)
r = cl.api("/api/models/load", {"model_name": cl.MODEL})
if "_http_error" in r:
    print("加载失败", r["_body"][:150]); raise SystemExit(1)
v = cl.api("/api/models/vision", {"model_name": cl.MODEL, "enabled": False})
print("关视觉", "ok" if "status" in v else v)
it = cl.resident() or {}
print("上线 ngl", it.get("gpu_layers"), "mmproj", repr(it.get("mmproj")),
      "空闲", json.dumps(it.get("host_free")))
try:
    low = cl.send_prompts()
except Exception as e:
    print("请求阶段挂了：", str(e)[:180]); low = None
print("请求中最低空闲", low, "MB")
