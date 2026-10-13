# LM Loader

**桌面端 AI 模型加载器** — 平替 Ollama + LM Studio，集成三进制 llama-server 引擎

## 快速开始

### 1. 安装

```bash
git clone https://gitee.com/buyuhuayouluo/lmloader.git
cd lmloader
pip install -e .
```

要求 Python 3.10+。

### 2. 配置

```bash
cp config.yaml.example config.yaml
```

`config.yaml` 中引擎路径留空即可，首次启动后在**设置页**点「一键更新」自动下载引擎。

### 3. 启动 GUI

```bash
lmloader-gui
```

桌面窗口包含聊天、模型管理、下载中心、资源监控、设置等完整功能。

### 4. 下载引擎（首次使用）

打开 GUI → 设置 → 引擎区域 → 点「检查更新」→「一键更新」。
程序会自动检测 CUDA 运行时，缺失时一并下载 cudart 捆绑包。

### 5. 下载并加载模型

- GUI 下载中心搜索 HuggingFace GGUF 模型，或直接输入 `repo:tag`
- 模型管理页点「加载」即可开始对话

---

## 产品定位

LM Loader 是一个**独立的桌面模型管理工具**，不是任何外部服务的客户端或路由中转：

- **替代 Ollama**：提供模型下载、仓库管理、Ollama 兼容 API、守护进程常驻
- **替代 LM Studio**：提供图形化聊天工作台、模型下载中心、设置面板、资源监控
- **集成三进制引擎**：内置 llama-server（含三进制版），在同一个工具内完成 模型下载 → 加载 → 对话 → 卸载 全流程

## 核心特性

### 引擎层
- **双引擎注册表**：primary 引擎（标准 GGUF）+ ternary 引擎（三进制模型）
- **自动路由**：按 GGUF 量化类型自动选择引擎（PQ2_0_TERNARY/TL1/TL2 → ternary 引擎）
- **守护进程**：服务常驻，优雅启停
- **端口管理**：自动分配可用端口

### 模型管理
- **本地扫描**：自动发现目录中的 GGUF 模型
- **远程下载**：从 HuggingFace 下载模型（支持国内镜像、断点续传）
- **版本管理**：支持 `name:tag` 语义，模型库持久化到 SQLite
- **一键操作**：加载/卸载/切换，实时显示引擎、端口、PID

### API 兼容
- **OpenAI 兼容**：`/v1/models`、`/v1/chat/completions`、`/v1/completions`
- **Ollama 兼容**：`/api/tags`、`/api/generate`、`/api/chat`、`/api/pull`、`/api/delete`
- **流式支持**：所有聊天接口支持 SSE 流式输出

### 前端界面
- **聊天工作台**：会话管理、流式渲染、Markdown 格式转换
- **模型管理**：卡片视图、加载状态、引擎信息
- **下载中心**：HuggingFace 搜索、进度条、断点续传
- **设置面板**：引擎路径、GPU 层数、上下文大小、KV cache 等参数配置
- **资源监控**：GPU/CPU/内存实时曲线、吞吐量统计

## 安装

```bash
cd lmloader
pip install -e .
```

## 配置

编辑 `config.yaml`：

```yaml
# 模型扫描目录
model_dirs:
  - ./models

# 引擎注册表（留空则使用内置引擎，或在 GUI 设置页一键更新）
engines:
  primary:
    binary_path: ""
    defaults:
      n_gpu_layers: 999
      ctx_size: 65536
      cache_type_k: q4_0
      cache_type_v: q4_0
      flash_attn: true
      threads: 8
  ternary:
    binary_path: ""
    defaults:
      n_gpu_layers: 999
      ctx_size: 32768

# CUDA 路径 (可选，留空自动检测)
cuda_path: ""

# 端口范围
port_range:
  start: 8100
  end: 8200

# 下载配置
download:
  hf_mirror: https://hf-mirror.com
  chunk_size: 10485760  # 10MB
  max_retries: 3

# 服务配置
server:
  host: 127.0.0.1
  port: 8900

# 数据库
database:
  path: ./lmloader.db

# 日志
logging:
  level: INFO
  file: ./lmloader.log
```

## 使用

### 启动守护服务

```bash
lmcli serve
```

服务启动后常驻后台，浏览器打开 http://localhost:8900 使用 Web 界面。

### 命令行

```bash
# 下载模型 (从 HuggingFace)
lmcli pull Qwen/Qwen3-4B-GGUF:Q4_K_M

# 列出模型库
lmcli list

# 加载模型 (服务未运行时自动拉起)
lmcli run Qwen/Qwen3-4B-GGUF:Q4_K_M

# 查看已加载模型
lmcli ps

# 切换模型
lmcli switch another-model:tag

# 卸载模型
lmcli unload model-name

# 停止守护服务
lmcli stop

# 查看引擎状态
lmcli status
lmcli engine-info
```

### API 调用

**OpenAI 兼容：**

```bash
curl http://localhost:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-GGUF:Q4_K_M",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

**Ollama 兼容：**

```bash
curl http://localhost:8900/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-GGUF:Q4_K_M",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

## 架构

```
┌──────────────────────────────────────────────────────┐
│ 前端 (完整 SPA)                                       │
│  聊天工作台 · 模型管理 · 下载中心 · 设置面板 · 监控      │
├──────────────────────────────────────────────────────┤
│ API 层 (双轨兼容)                                     │
│  OpenAI 兼容 /v1/*  +  Ollama 兼容 /api/*  +  自有管理  │
├──────────────────────────────────────────────────────┤
│ 应用服务层                                            │
│  模型注册表 (SQLite) · 会话管理 · 配置中心               │
├──────────────────────────────────────────────────────┤
│ 引擎层 (核心：引擎注册表 + 自动选择)                  │
│  引擎注册表: primary 引擎 + ternary 引擎               │
│  选择器: 按 GGUF 量化类型/架构 路由到正确引擎          │
│  进程管理: 守护常驻 · 优雅启停 · 端口分配                  │
├──────────────────────────────────────────────────────┤
│ 基础设施层                                            │
│  GGUF 解析 · 下载器 (HF/镜像/断点续传) · 硬件探测      │
│  · 日志 · SQLite                                      │
└──────────────────────────────────────────────────────┘
```

## 引擎路由规则

| 量化类型 | 引擎 |
|---------|------|
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ2_XXS, IQ3_XXS 等标准类型 | primary |
| PQ2_0_TERNARY, TL1_TERNARY, TL2_TERNARY, TQ1_0, TQ2_0, I2_S, TL1, TL2 | ternary |

路由结果在前端和 CLI 中明确展示，用户随时知道模型跑在哪个引擎上。

## 开发

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 格式化代码
black lmloader/
```

## 测试验证

| 测试项 | 结果 |
|--------|------|
| GUI 启动（PySide6 + API 服务） | 通过 |
| 双引擎检测（primary b11552 + ternary prism-stable，CUDA 可用） | 通过 |
| 模型扫描（9 个 GGUF 模型） | 通过 |
| 模型加载（llama3.2 3B → 端口 8100，GPU 999 层） | 通过 |
| 对话推理（OpenAI 兼容 API） | 通过 |
| 模型卸载 | 通过 |
| ConfigGuardian 配置审批锁 | 通过 |
| 引擎一键回退 | 通过 |
| CUDA 运行时自动检测与捆绑 | 通过 |

## License

MIT
