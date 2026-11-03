# 第二阶段：视频时间范围、持久化缓存与统一工具

## 1. 实现范围

本阶段保留现有 RDV，新增以下能力：

- 指定原视频绝对时间范围分析：`start_time`、`end_time`、
  `sample_interval`、`max_frames`；
- 负数、倒置范围、视频越界、非法类型和无穷数检查；
- 每个采样点保存请求时间、实际帧时间、帧号、全部保留三元组及分数；
- 未预采样时间点可读取对应帧补充识别，记录 `requested_time` 与
  `actual_frame_time`，不使用邻近采样结果冒充精确观察；
- 区分 `recognized_prediction`、`uncertain_prediction`、
  `below_threshold_candidate`、`model_no_recognition`；
- 模型未识别到明确写为证据不足，不推断“确定没有操作”；
- 预测事件组保留所有采样证据和来源，并标注为非人工真值、非真实手术阶段；
- 视频结果持久化并可跨进程重载；
- 离线视频自然语言问答和模拟实时模式；
- 固定、验证后的工具名称、参数和结构化错误。

主要代码位置：

```text
tools/video_tool.py          视频分析、补帧、查询、缓存和重载
agent/video_qa.py            离线/模拟实时确定性问答
agent/tool_registry.py       固定工具 schema、参数验证和执行入口
agent/query_router.py        时间点、范围、当前、过去N秒解析
agent/llm_agent.py           Agent 接入固定工具和离线视频路径
tools/run_video_qa.py        用户运行入口
tools/stage2_smoke.py        真实 RDV 小样本复现脚本
tests/test_video_stage2.py   第二阶段自动测试
```

## 2. 缓存身份与持久化

缓存键包含：

```text
视频内容 SHA-256
+ RDV 模型版本
+ checkpoint SHA-256
+ component/triplet 阈值
+ top-k 配置
+ 分析开始/结束时间
+ 抽帧间隔
+ max_frames
+ 结果 schema 版本
```

结果保存到：

```text
artifacts/cache/videos/<cache-key>.json
```

结果格式为 `surgical-video-analysis/v2`。`video.video_id` 由视频内容哈希生成，
不会仅按文件名复用。修改模型、阈值、采样配置或视频内容都会产生不同缓存键。

## 3. 固定工具接口

统一接口版本为 `surgical-agent-tools/v1`：

| 工具 | 必要参数 | 用途 |
|---|---|---|
| `analyze_video` | `video_path` | 分析指定原视频绝对时间范围并持久化 |
| `query_video_time` | `video_id`, `time` | 查询时间点，必要时补帧 |
| `query_video_range` | `video_id`, `start_time`, `end_time` | 返回范围内采样预测和事件组 |
| `summarize_video_events` | `video_id` | 汇总范围内预测操作与证据时间 |
| `query_surgical_knowledge` | 至少一个查询字段 | 查询已有可追溯知识文件 |

所有接口先做名称、必要参数、额外参数和 JSON 类型验证。错误统一返回：

```json
{
  "ok": false,
  "status": "error",
  "tool_api_version": "surgical-agent-tools/v1",
  "error": {
    "code": "...",
    "message": "...",
    "details": {}
  }
}
```

旧的 `analyze_surgical_video` 和 `query_surgical_timeline` 暂时保留为兼容层，
API 大模型模式未删除。

## 4. 离线问答入口

范围分析：

```powershell
conda activate rdv4
python -m tools.run_video_qa `
  --video test.mp4 `
  --question "分析第10秒到第30秒。" `
  --sample-interval 2
```

指定时刻：

```powershell
python -m tools.run_video_qa `
  --video test.mp4 `
  --question "第12秒使用了什么器械？"
```

模拟实时模式：

```powershell
python -m tools.run_video_qa `
  --video test.mp4 `
  --question "过去20秒发生了什么？" `
  --reference-time 30 `
  --realtime
```

`当前`、`现在`、`过去N秒` 在新接口中必须由程序提供 `reference_time`。
实时模式只返回 `reference_time` 及以前的样本；未来查询返回
`REALTIME_FUTURE_ACCESS`。

`SurgicalLLMAgent.answer()` 同样新增了可选参数：

```python
agent.answer(
    "过去20秒发生了什么？",
    video_path="test.mp4",
    reference_time=30.0,
    simulated_realtime=True,
)
```

基础视频观察问题由程序基于工具结构化结果直接渲染，日志路径为
`program_template`。需要解释或知识组合的问题仍可进入 API 大模型路径。

## 5. 验证命令与结果

自动测试：

```powershell
conda run -n rdv4 python -m pytest -q tests `
  test_confidence_policy.py test_memory_selection.py `
  test_temporal_routing.py --ignore=tests/test_deepseek_api.py
```

结果：`28 passed`。

真实 RDV 小样本：

```powershell
conda run -n rdv4 python -m tools.stage2_smoke `
  --video test.mp4 `
  --output artifacts/stage2_smoke.json `
  --force
```

真实运行结果：

- 分析原视频 10～14 秒，采样 10、12、14 秒；
- 三个采样点均没有三元组达到当前 `0.3` 保留阈值；
- 仍保存了最高候选及分数，并报告 `below_threshold_candidate`；
- 查询未预采样的 11 秒时实际读取帧 275、实际时间 11.000 秒；
- 第一次 11 秒查询执行补帧，第二次查询复用持久化结果；
- 同配置再次分析命中缓存；
- 实时边界为 12 秒时查询 13 秒，正确返回
  `REALTIME_FUTURE_ACCESS`；
- 另一次“过去20秒”、参考时刻 14 秒的真实运行仅返回 0～14 秒样本，
  其中 0～8 秒采样点预测到 `hook - dissect - cystic_duct`；
- 这些是项目测试视频上的 RDV 预测，不是人工真值或效果指标。

完整结果见 `artifacts/stage2_smoke.json`，每题追踪见
`artifacts/logs/agent_audit.jsonl`。

## 6. 本阶段仍缺少的外部文件

- CholecT 实际人工标注目录、版本说明和标注格式样例；
- 用于后续问答测试的独立视频划分；
- 真实说话人语音文件及人工转写；
- 当前 Codex 子进程可访问的 DeepSeek API Key（不影响离线视频问答）。

因此本阶段没有报告数据集 RDV mAP，也没有把 RDV 预测作为标准答案。

