# 阶段5：真实基线与本地LoRA冒烟验证

> 后续数值稳定性和长证据修复见`docs/phase5_pretraining_fixes.md`。其中修复后的全新adapter与审计结果取代本文件中存在NaN告警的旧5步adapter，旧产物仅保留为历史复现证据。

## 1. 本阶段结论

本阶段没有修改RDV网络、权重、CholecT50原始标注或冻结测试集，也没有启动正式完整训练。驱动更新后，独立` surgical-llm `环境已经完成真实CUDA运算、Qwen2.5-0.5B五步LoRA训练、独立验证、adapter保存与重载，以及一次生成—工具解析—工具执行—二次生成检查。

五步训练证明流程可运行，不证明模型已经学会任务。工具闭环的程序路径运行成功，但该次模型选择了错误工具，且最终回答缺少规定的结构化字段，因此任务闭环没有成功。

## 2. CUDA与独立环境

当前报告：`artifacts/stage5/environment_after_driver_update.json`与`artifacts/stage5/cuda_runtime_after_driver_update.json`。升级前失败记录`cuda_runtime_before_llm_deps.json`保留为历史，不再代表当前状态。

- GPU：RTX 2060 6GB；驱动621.01。
- Python：3.10.18；PyTorch 1.13.0+cu117；`torch.version.cuda=11.7`。
- CUDA实测：标量张量分配、1024×1024矩阵乘法、反向传播和主机回传均通过；梯度有限，校验误差为0。
- Transformers 4.46.3、PEFT 0.13.2、Accelerate 1.1.1均已真实导入并用于训练。其声明最低版本分别与当前PyTorch兼容；五步训练进一步验证了模型加载、PEFT注入、Trainer和Accelerate路径。
- `pip check`仍报告OpenCV 5要求NumPy≥2，而环境是NumPy 1.26.4。该冲突不在本次纯文字LoRA路径中，没有为它修改独立环境，更没有修改`rdv4`。

## 3. 训练证据压缩与全量检查

`compact_tool_context_v2`保留工具状态和错误、视频/时间/帧身份、全部三元组及其配对、全部事件边界和截断标志、发生截断时的原始边界、来源类型/数据集版本/人工真值标志。只移除不影响答案的重复诊断字段，例如人工真值置信度、重复实例计数、原始框、逐帧冗余和重复边界规则。标准化原始记录完全不变。

压缩事实一致性审计：训练7,610/7,610、验证1,100/1,100均匹配，无回答事实丢失。1024 token下采用整条过滤，不局部截断工具调用与工具返回。

| 类型 | 训练保留/总数 | 验证保留/总数 |
|---|---:|---:|
| instrument | 1400/1400 | 200/200 |
| action | 1400/1400 | 200/200 |
| target | 1400/1400 | 200/200 |
| triplet | 1157/1400 | 176/200 |
| time_point_operation | 1157/1400 | 176/200 |
| time_range_summary | 37/280 | 4/40 |
| continuous_event | 33/280 | 3/40 |
| 10类异常样本合计 | 50/50 | 20/20 |
| 总计 | 6634/7610 | 979/1100 |

loss mask全量检查通过：system/user/tool监督token均为0，只有assistant工具调用和最终回答参与监督。过滤的976条训练和121条验证样本仍保留在原始数据与审计清单中。正式训练前应进一步改善范围/事件样本占比，不能把当前过滤结果误当作最终数据方案。

## 4. 五步LoRA真实运行

配置：FP16 LoRA，序列上限1024，批量1，梯度累积4，gradient checkpointing开启，LoRA r=8/alpha=16，目标层为q/k/v/o projection。分层选取16条训练样本，覆盖全部七类正常问题和九类异常问题；验证使用4条独立验证样本。

- 实际步数：5；完整训练部分10.75秒，0.465 optimizer step/s。
- 训练loss：1.8777；验证loss：0.8262。两者只用于流程检查，不能解释为性能提升。
- 峰值实际张量分配：2674.34 MiB。
- PyTorch/WDDM峰值保留地址空间：8190 MiB；这是缓存分配器/虚拟保留，不是8GB物理显存占用，不能与6GB物理显存直接比较。
- adapter已保存并重载；权重文件4,350,392字节。
- 数值告警：第1、2步`grad_norm=NaN`，第3至5步恢复为有限值；训练和验证loss均有限。该问题归为FP16数值稳定性告警，不是OOM或依赖错误。正式训练前必须用更低学习率再做稳定性验证。

## 5. 工具闭环结果

adapter重载后，模型成功生成可解析的Qwen `<tool_call>`。参考要求`query_video_time`，模型却生成`query_surgical_knowledge`；参数通过schema校验，但人工标注模拟器找不到对应冻结证据并返回`ANNOTATION_SIMULATOR_MISS`。模型随后生成了用户可见的错误说明，但未输出`STRUCTURED_JSON`，解析结果为`structured_marker_missing`。

因此本次结论是：工具调用生成、解析、校验、执行、工具结果回填和二次生成的程序链路均实际运行；模型任务选择和最终结构化格式仍失败。五步冒烟不用于评价学习效果。

## 6. 基线、API与测试状态

- 本地微调前真实基础模型＋人工标注模拟证据：7条分层验证样本，工具选择0/7、联合工具参数0/7、自动结构化答案覆盖0/7；不是RDV端到端结果。
- 当前进程没有`DEEPSEEK_API_KEY`，本阶段没有执行付费API真实基线，也没有把不可用请求计为模型0分。
- 完整普通测试：`50 passed, 3 skipped`。跳过项是显式控制的外部API测试。
- 没有使用真实RDV证据评价本次文字模型，也没有运行冻结测试集。

## 7. 产物

- CUDA：`artifacts/stage5/cuda_runtime_after_driver_update.json`
- 环境：`artifacts/stage5/environment_after_driver_update.json`
- token审计：`tokenized_train_audit_v2.json`、`tokenized_validation_audit_v2.json`
- 压缩审计：`training_compaction_audit.json`、`validation_compaction_audit.json`
- 完整训练日志：`artifacts/stage5/lora_smoke_after_driver_update/smoke_report.json`
- 紧凑运行摘要：`artifacts/stage5/lora_smoke_after_driver_update/smoke_summary.json`
- adapter：`artifacts/stage5/lora_smoke_after_driver_update/adapter/`
- 正式训练草案：`config/local_lora_formal_draft.json`，默认未授权执行。

## 8. 可复现命令

```powershell
conda activate surgical-llm
$env:HF_HOME = (Resolve-Path artifacts).Path + '\hf_cache'
python tools/verify_cuda_runtime.py --output artifacts/stage5/cuda_runtime_after_driver_update.json
python tools/audit_tokenized_training.py --model models/Qwen2.5-0.5B-Instruct --input artifacts/local_training/train.qwen_tools.jsonl --max-length 1024 --output artifacts/stage5/tokenized_train_audit_v2.json
python tools/smoke_train_local.py --train-data artifacts/local_training/train.qwen_tools.jsonl --validation-data artifacts/local_training/validation.qwen_tools.jsonl --output-dir artifacts/stage5/lora_smoke_after_driver_update
```

不要直接执行正式训练草案。下一步应先降低学习率复验FP16梯度稳定性，并改善范围/事件类样本在1024 token下的保留率。
