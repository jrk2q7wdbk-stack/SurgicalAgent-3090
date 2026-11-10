# 阶段5：Qwen 检查点路径修复与断点恢复验证

## 1. 历史实验保留

失败实验 `artifacts/stage6/formal_round1/` 未删除、未覆盖。`run_report.json` 仍为 `status=failed`；第 200 步 adapter 的 SHA-256 仍为：

`54b91c8847deac3efc968fd4d450d7b3103ca767c2e594b4a3648f5d6580f660`

该 adapter 已使用原 Qwen2.5-0.5B 基础模型实际加载成功。192 个 LoRA 参数张量、1,081,344 个参数值全部有限。由于当时没有写成 `training_state.pt`，缺少 optimizer、scheduler、AMP scaler、global step 和 RNG 状态，因此分类固定为“诊断用部分训练权重”，不是可可靠续训的完整检查点。诊断结果位于：

`artifacts/stage6/diagnostics/checkpoint_step_000200_adapter.json`

## 2. 根因

- 工作目录是原项目目录 `D:/桌面/SurgicalAgent/SurgicalAgent_v3`，没有被移动。
- 正式训练脚本没有创建训练子进程；外层仅由 `conda run` 启动一次 Python，参数没有把输出路径改成其他目录。
- 原配置的相对输出目录经 `ROOT / output_dir` 解析后成为含中文的绝对路径。
- `checkpoint.mkdir(...)` 成功，`model.save_pretrained(.../adapter)` 也成功。
- 紧接着的 `torch.save(payload, Path(.../training_state.pt))` 报父目录不存在。这把故障限定在 PyTorch 1.13 Windows 文件名接口，而不是目录没有创建。
- 日志中曾看到的“妗岄潰”是 Windows PowerShell 读取无 BOM UTF-8 文件时的显示乱码；使用 `Get-Content -Encoding UTF8` 可看到路径实际仍为“桌面”。

修复采用双保险：训练输出强制使用纯英文绝对路径；`torch.save/torch.load` 通过 Python 二进制文件句柄调用，不再把 Unicode 文件名交给旧版 PyTorch C++ writer。

## 3. 检查点内容和恢复规则

`tools/train_lora_round1.py` 现在保存：

- LoRA adapter；
- optimizer state；
- scheduler state；
- AMP GradScaler state；
- epoch、epoch_position、micro_step、optimizer_step、successful_optimizer_steps；
- Python、PyTorch CPU/CUDA 和 NumPy RNG state；
- `state.json` 与完成标记 `checkpoint_manifest.json`。

只有包含上述完整 v2 结构且 manifest 标记 `complete=true` 的检查点可传给 `--resume-from-checkpoint`。旧第 200 步 adapter 会被拒绝为续训来源。

## 4. 实际短训练验证

输出目录：`D:/SurgicalAgentRuns/qwen_checkpoint_smoke_20261103`

先从原始基础模型训练到 optimizer step 2，再重新加载完整检查点并训练到 step 4。验证结果：

- step 2：micro step 16，successful optimizer steps 2，scheduler last_epoch 2，LR `7.042253521126761e-07`；
- step 4：micro step 32，successful optimizer steps 4，scheduler last_epoch 4，LR `1.4084507042253521e-06`；
- 两段训练均无跳过更新，梯度有限，AMP scale 均为 8192；
- optimizer 的 576 个状态张量全部有限；
- 192/192 个 adapter 张量在恢复后的两个更新步中发生变化并保持有限；
- Python、NumPy、PyTorch CPU/CUDA RNG 状态齐全；
- 第 4 步峰值 allocated 显存约 3222 MiB；首次第 1→2 步峰值约 3891 MiB；
- `test_split_used=false`，未运行冻结测试集，也未进行正式全量训练。

机器可判定报告：`artifacts/stage6/diagnostics/checkpoint_resume_verification.json`，其中 `valid=true`。外部完整日志和检查点保存在英文目录中。

回归测试：`23 passed`，覆盖 Unicode 文件句柄保存、绝对路径解析、旧部分 adapter 不可作为完整断点，以及阶段4评测相关测试。

GPU 检查未发现本次失败训练残留的 compute process，因此没有关闭任何 Python 进程。

## 5. 正式训练配置与命令（尚未执行）

新配置为 `config/local_lora_round1_ascii.json`，输出到独立新目录 `D:/SurgicalAgentRuns/qwen_round1_fresh_20261103`。命令不带 `--resume-from-checkpoint`，因此会从原始 Qwen 基础模型重新初始化 LoRA：

```powershell
conda activate surgical-llm
Set-Location -LiteralPath "D:\桌面\SurgicalAgent\SurgicalAgent_v3"
$env:HF_HOME = "D:\SurgicalAgentRuns\hf_cache"
python tools/train_lora_round1.py --config config/local_lora_round1_ascii.json --confirm-full-training
```

该命令当前只作为后续正式运行入口，本次未执行。
