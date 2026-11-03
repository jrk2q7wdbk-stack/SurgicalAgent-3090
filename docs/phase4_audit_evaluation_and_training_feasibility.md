# 阶段4：数据复核、异常样本、自动评测与训练可行性

## 1. 本阶段边界

本阶段没有修改 RDV 网络、权重或 CholecT50 原始标注，没有重新划分视频，也没有启动正式微调。原始测试集的 SHA-256 被记录；异常测试为单独冻结补充集。验证集用于协议冒烟，测试集未用于提示词、阈值或检查点选择。

## 2. 数据审计

入口：`tools/audit_training_data.py`。产物：

- `artifacts/data_audit/cholect50_rdv_original/data_audit_report.json`
- `artifacts/data_audit/cholect50_rdv_original/data_provenance.json`
- `artifacts/data_audit/cholect50_rdv_original/human_review_checklist.jsonl`

实际复核结果：10,800 条；train/validation/test 为 7,560/1,080/2,160；七类总分布分别为五类点查询各 2,000，两类范围/事件各 400；可回答 10,708，证据不足 92。视频集合互斥，sample ID 无冲突，点查询保存的全部三元组与原标注一致，审计 `valid=true`、`issues=[]`。

时间戳规则通过真实标注键与 PNG 文件名复核：键 `0/1/末帧` 对应 `000000.png/000001.png/末帧.png`，Release 2.0 的 `fps=1.0`，代码使用零基帧 `frame_id/fps`。未发现生成数据中的一秒偏移。数据集没有独立的原始视频采集 timecode，因此不能从这些文件进一步证明“采集绝对时间”的语义。

人工复核清单共 148 条：包含全部 92 条证据不足样本，并以固定种子 20261102 分层补充七类问题、多三元组、罕见组合、范围查询和截断事件。所有 `reviewer_status` 均为 `pending`，程序没有填写人工通过。

RDV checkpoint 已记录文件哈希，但它只含 state dict，没有可验证的训练划分元数据。因此结论固定为：`training_video_scope_status=unknown`，不能宣称 RDV 权重与当前测试视频无泄漏。

## 3. 异常样本

入口：`tools/generate_abnormal_samples.py`。默认 train/validation/test_supplement 为 50/20/10 条，每种异常分别为 5/2/1 条。覆盖负时间、超长、反向范围、缺视频、缺时间、指代不明、视频读取失败、RDV 失败、空记录/证据不足和不支持问题。

行为严格分为：

- `refuse`：参数无效、证据不足或超范围能力；
- `clarify`：缺视频、缺时间或指代不明确，不调用工具；
- `report_tool_error`：工具已调用但失败，如实报告错误，不伪造正常证据。

`source_state` 区分 `unannotated`、`annotation_has_no_positive_triplet`、`rdv_empty`、`tool_error` 和非法参数。空证据不会被改写为“确定没有操作”。语义参数校验已加入统一工具注册表，API 与本地模型共用。

## 4. 评测定义

入口：`tools/run_evaluation.py`；逐题、汇总与人工复核分别保存为 `per_sample.jsonl`、`summary.json`、`manual_review.jsonl`。

- 工具选择准确率：分母是参考要求调用工具的题；实际工具在允许方案中为正确。漏调用和解析失败为错。
- 参数准确率：分母是工具选择正确的调用；全部必要参数同时正确才通过，并报告 `missing:<参数>`、`wrong:<参数>`。
- 联合准确率：分母是全部需调用工具的题；工具和必要参数都正确才通过。
- 完整答案准确率：必要结构化事实集合相同、状态相同、事件边界相同，且自然文字不与结构化答案冲突。不是关键词命中评分。
- 实体与三元组 P/R/F1：大小写与空白标准化、集合顺序无关；三元组按 `(器械,动作,靶点)` 整体匹配，不能拆散重配。micro 汇总 TP/FP/FN，macro 为逐题平均。参考与预测均为空记 1，一侧为空记 0。
- 同义词：仅使用版本化映射；默认含 `grasping→grasp`、`retracting→retract`、`gall bladder→gallbladder` 等。不得用参考答案补全解析失败的输出。
- 时间参数容差：默认 0.01 秒，可由 CLI 修改；事件边界当前要求标准化值相等。
- 异常正确率：分别以 `refuse/clarify/report_tool_error` 为分母，实际行为必须与参考一致。
- 正确拒答、错误拒答、覆盖率：仅纳入 `human_confirmed` 的可回答性标签；`pending` 单列排除，不冒充人工真值。
- 完整任务成功率：超时、格式错误、工具失败、超过轮数均为失败，并要求路由、参数和答案事实正确。
- 延迟：完整回答记录端到端耗时；非流式后端的首次用户可见输出为 unavailable。工具调用 token 不计首次答案。

自然语言没有结构化输出或文字与结构化事实冲突时进入 `manual_review.jsonl`。可选 LLM judge 尚未启用，未来也只作为辅助列。

## 5. 三条基线

`mock` 是参考协议 oracle，只验证评测链路，不是模型性能。`current-agent` 保留当前程序时间路由与模板回答，并用人工标注模拟工具排除 RDV 误差；输出明确标记 `program_template` 和 `mock=true`。`deepseek-autonomous` 绕过规则替模型选工具的路径，但仍执行统一参数校验、真实工具安全边界、超时、指数退避重试和最大调用轮数。`local` 是后续基础模型/微调模型的同协议接口。

已实际运行 35 条验证集分层协议 mock、35 条当前规则 Agent＋人工标注工具 mock，以及 20 条异常协议 mock。所有 1.0 仅表示 oracle/协议链路正确，`model_performance_eligible=false`，不是大模型效果。当前进程没有 `DEEPSEEK_API_KEY`，因此 DeepSeek 请求没有执行；不可用结果需标记 `run_status=external_dependency_unavailable`，不得当成模型 0 分或真实性能。

完整普通测试结果为 `45 passed, 3 skipped`。三项付费 DeepSeek 外部测试默认跳过；只有显式设置 `RUN_DEEPSEEK_API_TESTS=1` 才会纳入 pytest。

## 6. 本地训练可行性

现场结果见 `artifacts/training_feasibility/environment.json`：Windows、Python 3.10.18、PyTorch 1.13.0+cu117、RTX 2060 6 GB、驱动 457.85；nvidia-smi 检查时空闲约 4.9 GB；内存约 15.9 GB、可用约 4.5 GB；D 盘空闲约 103 GB。`transformers/peft/accelerate/trl/bitsandbytes/datasets/sentencepiece` 均未安装，选定模型也不在缓存中。

小显存首选验证模型为 `Qwen/Qwen2.5-0.5B-Instruct`：官方模型卡标明 Apache-2.0、0.49B 参数、24 层，模型 safetensors 约 988 MB；官方 Qwen 文档说明 Qwen2.5 tokenizer 模板支持 Hermes 风格 tool use，Transformers tool-use API 需高于 4.42。配置先做 FP16 LoRA，而不先依赖 Windows 上尚未验证的 bitsandbytes/QLoRA。6 GB 是否足够仍须以真实 smoke 峰值为准，文档不预先宣称成功。

训练转换已生成 7,610 条 train 和 1,100 条 validation 对话，0 拒绝。`training/tool_call_data.py` 只监督 assistant 的工具调用与最终答案；system/user/tool 均为 `-100`，tool 结果仅作上下文。超过 `max_length` 的整条样本被记录并过滤，不局部截断，因此不会拆开调用与结果。

`tools/smoke_train_local.py` 限制最多 5 步，包含验证、adapter 保存、重载和一次生成，并记录峰值显存与速度。当前真实执行停在 `blocked_missing_dependency: transformers`，没有模型加载、训练步、验证步、adapter 或性能数据。

官方依据：[Qwen2.5-0.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[Qwen2.5 Function Calling 文档](https://qwen.readthedocs.io/en/v2.5/framework/function_call.html)。

## 7. Windows PowerShell 命令

审计和评测（现有 `rdv4`）：

```powershell
conda activate rdv4
python tools/audit_training_data.py --dataset-root "E:\毕业设计\CholecT50"
python tools/generate_abnormal_samples.py
python tools/run_evaluation.py --input artifacts/training_data/cholect50_rdv_original/validation.standardized.jsonl --output-dir artifacts/evaluation/validation_protocol_mock --backend mock --limit 35
python tools/run_evaluation.py --input artifacts/training_data/cholect50_rdv_original/validation.standardized.jsonl --output-dir artifacts/evaluation/validation_current_agent_mock --backend current-agent --limit 35
```

小额 API 验证（只运行明确的验证集上限，不运行测试集）：

```powershell
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'DeepSeek API Key' -AsSecureString)).Password
python tools/run_evaluation.py --input artifacts/training_data/cholect50_rdv_original/validation.standardized.jsonl --output-dir artifacts/evaluation/validation_deepseek_35 --backend deepseek-autonomous --model deepseek-chat --limit 35 --timeout 45 --retries 2 --max-rounds 2
```

独立训练环境（命令已提供，但本阶段未执行安装或下载）：

```powershell
conda create -n surgical-llm python=3.10 -y
conda activate surgical-llm
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.3 accelerate==1.1.1 peft==0.13.2 safetensors
python tools/smoke_train_local.py --train-data artifacts/local_training/train.qwen_tools.jsonl --validation-data artifacts/local_training/validation.qwen_tools.jsonl --allow-download
```

`--allow-download` 会下载约 988 MB 权重及小型配置/tokenizer 文件；未明确授权前不运行。首次训练前应关闭占用 GPU 的 RDV/Python 进程，并用 `nvidia-smi` 再确认空闲显存。
