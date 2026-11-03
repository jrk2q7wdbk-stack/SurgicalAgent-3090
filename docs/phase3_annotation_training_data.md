# 人工标注工具调用训练数据阶段

语音阶段按用户要求暂缓。本阶段先完成文字大模型工具调用训练所需的人工标注数据。

## 数据集确认

实际数据根目录：

```text
E:\毕业设计\CholecT50\CholecT50
```

读取器严格限定为已验证的 CholecT50 Release 2.0 JSON 格式：

- 50 个原始视频；
- 100,863 个 1 FPS 标注帧；
- 150,927 个有效正三元组实例；
- 54,315 个多三元组帧；
- 11,061 个没有正三元组标签的帧；
- 每个原始实例为官方定义的 15 字段记录；
- JSON 声明帧数、标注键和 PNG 文件数全部一致。

没有正三元组标签的帧只用于“证据不足”样本，不解释为确定没有操作。

## 数据划分

默认使用官方代码中的 `cholect50` 原始 Rendezvous 划分：

- 训练：35 个视频；
- 验证：5 个视频；
- 测试：10 个视频。

三个集合按原始视频完全隔离。读取器还支持 `challenge` 和
`crossval` 划分。当前 RDV checkpoint 只包含 state dict，没有保存训练划分元数据；
因此 manifest 明确记录当前采用 `rdv_original`，不声称从 checkpoint 内恢复了划分。

## 已生成的问题类型

- 器械查询；
- 动作查询；
- 靶点查询；
- 完整三元组查询；
- 指定时间操作查询；
- 时间范围三元组汇总；
- 连续事件及片段边界截断查询。

连续事件仅在同一三元组出现在连续 1 FPS 人工标注帧时合并，不填补间隙。
事件越过查询窗口时分别标记 `left_truncated` 和 `right_truncated`。

## 样本格式

标准化记录包含：

```text
用户问题
正确工具调用及参数
工具返回
最终自然语言答案
结构化答案
视频编号和时间范围
证据来源
问题类型
是否可回答
训练/验证/测试划分
```

同一记录另行导出为五消息工具调用对话：

```text
system
user
assistant(tool_calls)
tool
assistant(final answer)
```

工具返回中的来源固定为 `human_annotation`、`is_ground_truth=true`。
RDV 预测没有参与标准答案生成。

## 生成结果

输出目录：

```text
artifacts/training_data/cholect50_rdv_original
```

样本数量：

| 划分 | 视频 | 样本 |
|---|---:|---:|
| train | 35 | 7,560 |
| validation | 5 | 1,080 |
| test | 10 | 2,160 |
| 合计 | 50 | 10,800 |

其中 92 条证据不足样本进入 `review_queue.jsonl`。生成器基于罕见标签组合进行
分层抽样，并限制同一视频连续重复标签；过滤决策保存在 `selection_log.jsonl`。
当前生成记录没有质量校验拒绝项，`rejected.jsonl` 为空。

## 输出文件

```text
manifest.json
train.standardized.jsonl
train.conversations.jsonl
validation.standardized.jsonl
validation.conversations.jsonl
test.standardized.jsonl
test.conversations.jsonl
selection_log.jsonl
review_queue.jsonl
rejected.jsonl
validation_report.json
```

## 复现命令

```powershell
conda activate rdv4
python -m tools.qa_generator `
  --dataset-root "E:\毕业设计\CholecT50" `
  --output-dir artifacts\training_data\cholect50_rdv_original `
  --split-variant rdv_original `
  --max-point-frames-per-video 40 `
  --max-per-signature 2 `
  --range-windows 10 30 `
  --range-stride 60 `
  --max-ranges-per-video 8 `
  --seed 2026
```

质量验证：

```powershell
python -m tools.validate_training_data `
  --input-dir artifacts\training_data\cholect50_rdv_original `
  --output artifacts\training_data\cholect50_rdv_original\validation_report.json
```

验证覆盖 JSON 可解析性、必要字段、工具参数与时间一致性、人工标注来源、
对话角色顺序、工具调用 ID、样本 ID 唯一性和视频划分泄漏。当前报告为
`valid=true`、`issues=[]`。

## 训练前仍需完成

- 当前 test 导出是独立视频上的模板样本，可用于结构检查，但不能代替最终的
  独立自然问法测试集；
- 尚未执行可选大模型问题改写和改写一致性过滤；
- 尚未加入参数缺失、工具失败、越界和需要澄清等专项训练样本；
- 尚未建立微调前基础模型评测基线；
- 尚未选择和下载基础文字模型。

本机 GPU 为 GeForce RTX 2060 6GB。该显存不适合全参数训练较大模型；后续应优先
考虑小型工具调用模型的 LoRA/QLoRA，并先进行显存小样本验证。

