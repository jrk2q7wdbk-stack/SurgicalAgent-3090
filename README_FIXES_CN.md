# SurgicalAgent v2：本次修订说明

最新的数据审计、异常样本、统一评测和本地 LoRA 可行性说明见
[`docs/phase4_audit_evaluation_and_training_feasibility.md`](docs/phase4_audit_evaluation_and_training_feasibility.md)。

## 1. RDV 结果与回答严格绑定

核心视觉问题不再让 LLM 自由改写标签。`SurgicalLLMAgent.answer()` 会直接读取
`best_triplet`，并由 `agent/confidence.py` 生成只包含 RDV 证据的回答。

统一置信度策略：

- `high`：`confidence >= 0.75`，正常回答；
- `medium`：`0.50 <= confidence < 0.75`，使用“可能、提示”等保守措辞；
- `low`：`confidence < 0.50`，拒绝确定判断，只报告最高候选供复核。

视觉输出中新增：`confidence_level`、`status`、`answer_policy`。

## 2. RDV 标签自动生成问题—回答对

从 100 个 IVT 标签生成 400 个可追溯问答对：

```powershell
python tools\qa_generator.py `
  --labels knowledge\triplet_classes.json `
  --output knowledge\triplet_qa.jsonl
```

单帧预测也可直接调用：

```python
from tools.qa_generator import generate_qa_pairs_from_prediction

pairs = generate_qa_pairs_from_prediction(
    vision_result,
    source_id="frame_000100",
    timestamp=4.0,
)
```

低置信度预测生成的样本会标记 `answerable=false`，答案明确拒绝确定判断。

## 3. 连续视频事件序列

`analyze_surgical_video()` 按时间间隔采样，返回带时间戳的
`event_sequence`。每个事件可能为：

- `action`：中/高置信度的完整三元组；
- `uncertain`：低置信度候选，不能作确定判断；
- `no_action`：没有完整有效动作标签。

`uncertain` 与 `no_action` 不再混为一类。事件中同时保留平均置信度、置信等级和
回答策略。视频末尾的待确认状态会在结束时正确收尾，临时帧也会自动清理。

## 4. 按问题检索时序记忆

`agent/query_router.py` 在 LLM 前先解析问题，再只读取对应的 Temporal Memory：

- “当前/现在” → 最新事件；
- “上一个事件” → 最近两个事件并选择前一个；
- “最近十秒” → 以最新事件时间为终点的十秒窗口；
- “20 秒”或“20 秒到 35 秒” → 精确时间点/范围；
- “最长、多少阶段、前后变化” → 对应统计或转换查询。

普通直接问题不会再携带完整聊天历史；只有“这个、刚才、上述”等指代问题才提供
最后一轮对话。视频完整时间线也不会自动塞给 LLM。

## 5. 运行验证

规则和记忆测试：

```powershell
python -m pytest -q tests
```

已有测试覆盖 0.30/0.50/0.75/0.95 边界、低置信度拒答、`uncertain` 与
`no_action` 区分、视频尾事件收尾、时间点/范围/最近十秒/上一个事件路由，以及
聊天历史最小化。
