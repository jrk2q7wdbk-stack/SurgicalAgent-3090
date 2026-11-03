# 第一阶段：项目检查与统一配置

## 1. 当前实际入口与依赖链

- `test_agent_capabilities.py`：现有最完整的能力演示入口，调用
  `agent.llm_agent.SurgicalLLMAgent`。
- `tests/test_video_llm_agent.py`：视频分析与后续时序问答演示。
- `test_inference.py`：单图 RDV 推理入口，调用
  `tools.vision_tool -> vision.inference -> vision.network.Rendezvous`。
- `tools/project_check.py`：第一阶段新增的只读项目自检入口。

主要依赖链：

```text
SurgicalLLMAgent
  -> tools/vision_tool.py -> RDV checkpoint + label JSON
  -> tools/video_tool.py  -> OpenCV + TemporalMemory + vision tool
  -> tools/knowledge_tool.py -> surgical_knowledge.json
  -> tools/graph_knowledge_tool.py -> surgical_knowledge_graph.json
  -> OpenAI-compatible client -> configured LLM service
```

项目当前没有生产级统一 CLI；文字/音频统一入口将在语音阶段实现。

## 2. 保留的视觉基线

当前仍使用 `vision.network.Rendezvous`（RDV），没有引入 Swin-T 或 SGRR。
默认权重为：

```text
models/rendezvous_l8_cholectcholect50_k0_batchnorm_lowres.pth
```

标签维度经 JSON 读取检查为：器械 6、动作 10、靶点 15、三元组 100。
权重文件存在不等于模型推理验证成功；必须在安装 PyTorch 后实际加载并推理。

## 3. 统一配置

默认配置为 `config/default.json`，用户覆盖示例为
`config/local.example.json`。可以通过以下环境变量指定覆盖文件：

```powershell
$env:SURGICAL_AGENT_CONFIG = "config/local.json"
```

也支持少量直接环境变量覆盖：

```powershell
$env:SURGICAL_MODEL_PATH = "D:\path\model.pth"
$env:SURGICAL_DEVICE = "cuda"
$env:SURGICAL_THRESHOLD = "0.5"
$env:SURGICAL_TRIPLET_THRESHOLD = "0.3"
$env:SURGICAL_SAMPLE_INTERVAL = "2.0"
$env:SURGICAL_LLM_BASE_URL = "https://api.deepseek.com"
$env:SURGICAL_LLM_MODEL = "deepseek-v4-flash"
```

API 密钥值不进入 JSON 配置。默认只从环境变量读取：

```powershell
$env:DEEPSEEK_API_KEY = "..."
```

自检和公开配置摘要只记录“是否设置”，不会记录密钥值。

## 4. 回答路径与日志

`SurgicalLLMAgent.answer()` 现在将每次请求追加到：

```text
artifacts/logs/agent_audit.jsonl
```

记录问题、路由、工具调用及参数、工具结果、最终答案、耗时，以及实际回答路径：

- `program_template`：程序模板/确定性渲染直接回答；
- `llm_generated`：大模型生成回答；
- `llm_unavailable`：问题需要大模型，但 API Key 未配置；
- `llm_error`：大模型 API 调用失败；
- `tool_round_limit`：达到工具调用轮次上限。

未配置 API Key 时不再阻止 Agent 构造；离线确定性路径仍可运行，需要大模型的
问题会明确返回缺失项。

## 5. 可复现检查命令

项目实际运行环境为 Conda `rdv4`：

```powershell
conda activate rdv4
```

该环境已确认具备 `torch`、`torchvision`、Pillow、NumPy、OpenCV、
OpenAI SDK 和 pytest。仓库内 `.venv` 不是本项目当前运行环境。

项目自检：

```powershell
conda run -n rdv4 python -m tools.project_check `
  --output artifacts/project_check_stage1_rdv4.json
```

依赖无关的第一阶段测试与语法检查：

```powershell
conda run -n rdv4 python -m unittest tests.test_config_stage1 -v
conda run -n rdv4 python -m compileall -q agent tools vision tests
```

安装依赖后再执行现有测试：

```powershell
conda run -n rdv4 python -m pytest -q tests `
  test_confidence_policy.py test_memory_selection.py `
  test_temporal_routing.py --ignore=tests/test_deepseek_api.py
```

## 6. 已确认阻塞与外部文件

当前项目树内：

- RDV 权重、标签文件和两个知识 JSON 均存在；
- 未发现文件名带 `(1)`、`(2)` 等附件重名后缀，因此没有执行覆盖或重命名；
- 未发现 CholecT 人工标注数据，标注读取器和训练数据构建不能做真实数据验证；
- 未发现语音识别模型或语音测试集；
- `rdv4` 环境的运行依赖齐全；
- 当前 Codex 子进程没有继承 `DEEPSEEK_API_KEY`，因此在线 API 测试尚未执行；
- 未提供基础文字模型、本地微调权重、GPU 信息和许可证确认，不能进行真实微调。

详细机器可读结果见 `artifacts/project_check_stage1_rdv4.json`。

## 7. 第一阶段真实运行结果

- 离线测试：18 项通过；在线 `tests/test_deepseek_api.py` 因当前子进程没有
  API Key 而单独排除。
- RDV checkpoint 在 CPU 上严格加载成功。
- `test.jpg` 真实推理返回两个保留三元组；最高结果为
  `grasper - grasp - gallbladder`，分数约 `0.9978`。
- Agent 程序模板路径真实运行成功，审计记录为
  `response_path=program_template`、`status=ok`。
- 上述结果仅是单张项目测试图片的小样本运行，不代表数据集指标或真实系统效果。

## 8. 后续阶段边界

下一阶段将在现有 RDV 基线上实现视频身份、时间范围、精确补帧、全量三元组保留、
持久化缓存、离线问答和模拟实时访问边界。人工标注生成、自动评测和大模型训练分别
保留为后续独立流水线，不把 RDV 预测当作人工真值。

