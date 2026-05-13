# VoxCPM2 与官方仓库同步状态

> 本文件用于记录本地仓库相对上游 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 的同步进度、差异点与下一步操作。
>
> 相关文档：
> - 本地自定义改动清单：[CUSTOM_CHANGES.md](./CUSTOM_CHANGES.md)
> - 升级前结构化备份：[backup_before_upgrade_20260429_154548/README.md](./backup_before_upgrade_20260429_154548/README.md)

---

## 最近一次扫描

| 项 | 值 |
|---|---|
| 扫描时间 | 2026-05-13 |
| 官方仓库 | https://github.com/OpenBMB/VoxCPM |
| 官方 main HEAD | `19b6bf7` - `fix: handle LoRA rank mismatch during inference in lora_ft_webui`，2026-04-28 |
| 官方最新 Tag | `2.0.3` - `19b6bf7` |
| 本地当前分支 | `local-custom` |
| 本地当前 HEAD | `c11c87c` - `Keep local artifacts out of git and save VoxCPM updates` |
| 官方自上次记录后的新增 commit | `0` |
| 当前结论 | 官方 `main` 暂无新提交；仅最新 tag 从 `2.0.2` 更新为 `2.0.3` |

---

## Git 对比结果

本次执行：

```powershell
git fetch upstream --prune
git log --oneline --decorate -n 8 upstream/main
git log --oneline --date=short --pretty=format:"%h %ad %s" 19b6bf7..upstream/main
git rev-list --left-right --count HEAD...upstream/main
```

结果说明：

- `upstream/main` 仍指向 `19b6bf7`。
- `19b6bf7..upstream/main` 没有输出，说明官方没有比上次记录更新的 commit。
- `git rev-list --left-right --count HEAD...upstream/main` 输出 `3 13`：
  - 左侧 `3` 是本地 `local-custom` 独有提交。
  - 右侧 `13` 是官方历史中未作为 Git merge 祖先进入本地分支的提交。
  - 注意：这些官方变更已经按文件内容人工对齐过一批，所以这里不能简单理解为“还有 13 个功能没合入”；它主要反映当前分支没有做完整的 upstream merge。

---

## 当前本地独有提交

| SHA | 日期 | 说明 |
|---|---|---|
| `c11c87c` | 2026-05-13 | 忽略本地模型/训练产物，并提交 VoxCPM 本地更新 |
| `7c0d686` | 2026-04-17 | 在 `CUSTOM_CHANGES.md` 增加部署说明 |
| `35934e9` | 2026-04-17 | 增加本地自定义功能、音频超分、批量生成、LoRA WebUI 与参考音频 |

---

## 已确认对齐的官方基线

上次对齐目标仍有效：

| 官方 commit | 日期 | 说明 |
|---|---|---|
| `19b6bf7` | 2026-04-28 | 修复 `lora_ft_webui.py` 推理时 LoRA rank mismatch 处理 |

该 commit 现在同时是官方 `main` 和最新 tag `2.0.3`。

---

## 本地持久差异

这些是本地刻意保留、不要被官方覆盖的改动：

- `lora_ft_webui.py`：本地 LoRA WebUI、分段生成、进度显示、checkpoint 加载等定制逻辑。
- `app.py`：本地 WebUI 调整、LoRA 热加载/切换、超分与降噪相关开关。
- `audio_sr.py`：AP-BWE 音频超分模块。
- `batch_generate_emotions.py` / `batch_voice_design.py`：批量生成与声音设计脚本。
- `check_env.py`：环境检查脚本。
- `启动VoxCPM2.bat` / `启动LoRA_WebUI.bat`：Windows 启动脚本。
- `CUSTOM_CHANGES.md` / `UPSTREAM_SYNC_STATUS.md`：本地维护文档。
- `reference_audios/` 与 `reference_audios.zip`：本地参考音频资源。

以下内容属于本地运行产物，已在 `.gitignore` 中排除，不应上传 Git：

- `models/`
- `lora/`
- `lora_train_data/`
- `ap_bwe_checkpoints/`
- `backup_before_upgrade_*/`
- `VoxCPM_*.pdf`
- `*.safetensors` / `*.pth` / `*.pt` / `*.ckpt`
- `runs/` / `wandb/` / `*.tfevents.*`

---

## 下一次同步步骤

当官方 `main` 再次更新时，建议按以下流程处理：

1. 扫描官方新增提交：

   ```powershell
   git fetch upstream --prune
   git log --oneline --date=short 19b6bf7..upstream/main
   ```

2. 逐条评估新增 commit 的风险：

   - 低风险：文档、测试、非冲突脚本，可以直接合入或 cherry-pick。
   - 中风险：`src/voxcpm/model/*`、`src/voxcpm/cli.py`、`src/voxcpm/training/*`，需要检查本地改动后合入。
   - 高风险：`app.py`、`lora_ft_webui.py`，只抽取明确 bug fix，避免整体覆盖本地定制。

3. 合入后验证：

   ```powershell
   python -m compileall app.py lora_ft_webui.py scripts src tests
   python -m pytest tests/test_validate.py tests/test_lora_checkpoint_loading.py scripts/test_pick_runtime_dtype.py
   ```

   当前机器的全局 Python 环境缺少 `pytest`，如需跑完整测试，需要先安装项目 `dev` 依赖或使用带 pytest 的虚拟环境。

4. 更新本文档的扫描时间、官方 HEAD、最新 tag、待合入 commit 清单与验证结果。

---

## 变更历史

| 日期 | 事件 |
|---|---|
| 2026-04-29 | 升级前结构化备份，基线为 `86bff0f` 附近 |
| 2026-05-06 | 完成一次完整拉取评估，追平到官方 `19b6bf7` |
| 2026-05-13 | 重新扫描官方仓库：无新增 commit；最新 tag 更新为 `2.0.3`；重写本文档为可读中文 |
