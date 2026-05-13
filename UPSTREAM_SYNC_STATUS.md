# 📊 VoxCPM2 与官方仓库同步状态

> **本文件用途**：追踪本地仓库相对于上游 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 的同步进度、差异点与合并历史。
> **相关文档**：
> - 本地自定义改动清单 → [`CUSTOM_CHANGES.md`](./CUSTOM_CHANGES.md)
> - 升级前结构化备份 → [`backup_before_upgrade_20260429_154548/README.md`](./backup_before_upgrade_20260429_154548/README.md)

---

## 🕒 最近一次扫描

| 项 | 值 |
|---|---|
| 扫描时间 | 2026-05-06 |
| 官方仓库 | https://github.com/OpenBMB/VoxCPM |
| 官方 main HEAD | `19b6bf7` — *fix: handle LoRA rank mismatch during inference in lora_ft_webui* （2026-04-28） |
| 官方最新 Tag | `v2.0.2` — `68af4fe`（2026-04-08） |
| **本地进度** | **已追平 main HEAD（`19b6bf7`）** |
| **待拉取 commit** | **0 条** ✅ |

---

## 📈 本地文件行数基线（验证依据）

| 文件 | 备份时本地（2026-04-29 前） | 备份时官方 | **当前本地** | 状态 |
|---|---|---|---|---|
| `src/voxcpm/model/voxcpm.py` | 993 | 1008 | **1006** | ✅ 已合并 |
| `src/voxcpm/model/voxcpm2.py` | 1244 | 1259 | **1257** | ✅ 已合并 |
| `src/voxcpm/cli.py` | 607 | 更多 | **653** | ✅ 已合并（新增 `cmd_validate`） |
| `src/voxcpm/training/validate.py` | ❌ 缺失 | 新增 | **310 行存在** | ✅ 已合并 |
| `tests/test_validate.py` | — | — | **252 行存在** | ✅ 已合并 |
| `lora_ft_webui.py` | 1635 | ~1700 | **1540** | ✅ 已合并（含 LoRA rank mismatch 修复） |
| `app.py` | 840 | ~500 | **513** | ✅ 已大幅对齐官方 |
| `scripts/train_voxcpm_finetune.py` | 844 | 844 | **840** | ✅ 基本一致 |

### 关键代码特征核对
- ✅ `lora_ft_webui.py` 含 `"LoRA rank mismatch (model r=..., checkpoint r=..., reloading...)"` → 合入 `19b6bf7` (2026-04-28)
- ✅ `cli.py` 266/309 行 `import soundfile as sf` 在函数体内 → 合入 `dd7b78f` (2026-04-24 defer imports)
- ✅ `cli.py` 290 行 `def cmd_validate(...)` → 合入 `4457617` (2026-04-12 validate CLI)
- ✅ `voxcpm2.py` 1082 行 "Yield only the newest patch latent for stateful VAE decode" → 合入 `3589598` (2026-04-15 stateful streaming VAE)
- ✅ `training/__init__.py` 导出 `validate_manifest, ValidationResult`

---

## 🔄 已合并的上游 commit（自 v2.0.2 以来 32 条）

### 🔧 代码修复与优化（18 条）
| SHA | 日期 | 说明 |
|---|---|---|
| `5611bd0` | 2026-04-08 | optim app.py |
| `6620513` | 2026-04-08 | perf: stateful streaming VAE decode — eliminate redundant overlap |
| `75cfa3e` | 2026-04-09 | fix: uncompiled feat_encoder for prefill（CUDA Graph 动态 shape 累积） |
| `4f4a5b9` | 2026-04-09 | fix: _generate() 类型检查顺序（非字符串输入 AttributeError） |
| `e4e0496` | 2026-04-11 | update finetuning pipeline and runtime device handling |
| `fb46aad` | 2026-04-11 | fix: close file handles in from_local() config loading |
| `b1584ae` | 2026-04-13 | fix: stabilize CPU SDPA mask broadcasting |
| `61b36d4` | 2026-04-13 | refactor: centralize generator cleanup in model helpers |
| `1565e83` | 2026-04-13 | fix: complete shared generator cleanup coverage |
| `38d61cd` | 2026-04-15 | fix(mps): force float32 on Apple Silicon |
| `f7f1b78` | 2026-04-15 | fix: correct transpose conv context |
| `3589598` | 2026-04-15 | Merge PR #212: stateful streaming VAE decode |
| `ec2acec` | 2026-04-17 | Harden LoRA checkpoint loading against untrusted pickle |
| `d3cc887` | 2026-04-21 | feat: enhance control text processing in VoxCPMDemo |
| `96d605b` | 2026-04-21 | fix(mps): align VOXCPM_MPS_DTYPE override set |
| `4509bec` | 2026-04-21 | fix: address four validation correctness issues |
| `dd7b78f` | 2026-04-24 | refactor(cli): defer soundfile/voxcpm.core imports |
| `19b6bf7` | **2026-04-28** | ⭐ fix: handle LoRA rank mismatch during inference in lora_ft_webui |

### ✨ 新功能（1 条）
| SHA | 日期 | 说明 |
|---|---|---|
| `4457617` | 2026-04-12 | feat: add `voxcpm validate` CLI for pre-flight training data checks（新增 `src/voxcpm/training/validate.py`） |

### 🧪 测试（1 条）
| SHA | 日期 | 说明 |
|---|---|---|
| `29577d5` | 2026-04-24 | test: fix test_cli_validate_exit_code |

### 🧹 杂项（1 条）
| SHA | 日期 | 说明 |
|---|---|---|
| `79c0cf6` | 2026-04-09 | chore: remove accidentally committed app_local.py |

### 📖 文档（4 条）
| SHA | 日期 | 说明 |
|---|---|---|
| `364eff6` | 2026-04-08 | update readme: python version |
| `6d10932` | 2026-04-08 | update readme |
| `eae0a29` | 2026-04-16 | docs: add ComfyUI RH link |
| `afa63e6` | 2026-04-17 | docs: add vLLM-Omni serving references |

### 🔀 Merge 节点（7 条，本身不带代码改动）
`abf01b9`、`5510503`、`13605c5`、`77f847f`、`a9b03a7`、`cd79a64`、`86bff0f`

---

## 🎨 本地相对官方的**持久性差异**

下列是本地刻意保留、**不会被官方覆盖**的定制点：

### 本地独有文件（官方没有）
- `audio_sr.py` — AP-BWE 音频超分辨率模块
- `batch_generate_emotions.py` — 批量情感语音生成
- `batch_voice_design.py` — 批量声音设计
- `check_env.py` — 环境检查
- `app_old.py` — 旧版 WebUI 备份
- `启动VoxCPM2.bat` / `启动LoRA_WebUI.bat` — Windows 一键启动脚本
- `lora_paper_zh.md` — LoRA 论文中文翻译
- `CUSTOM_CHANGES.md` / `UPSTREAM_SYNC_STATUS.md` — 文档
- `reference_audios/` / `lora_train_data/` — 本地数据
- `ap_bwe_checkpoints/` — 超分模型检查点

### 已二次定制的文件（深度改动）
- `lora_ft_webui.py`（1540 行）— 三种推理模式、分段生成、CSS 动画、进度条
- `app.py`（513 行）— LoRA 热加载/热交换、超分开关、降噪开关、dit_steps 滑块
  - 注：2026-04-16 已撤销"分段生成"逻辑，恢复整段一次性生成（详见 `CUSTOM_CHANGES.md`）

### 环境侧约定
- `torchcodec` 已卸载（参考音频降噪时会因缺失 FFmpeg DLL 崩溃）
- CUDA 推理使用 bfloat16
- 48kHz 下默认禁用 Denoiser/Enhance Audio（zipenhancer 会损伤高频）

---

## 🧭 下次同步操作清单

当上游 main 再更新时，建议按以下步骤：

1. **先扫描**：访问 https://github.com/OpenBMB/VoxCPM/commits/main 看 `19b6bf7` 之后是否有新提交
2. **做备份**：按 `backup_before_upgrade_YYYYMMDD_HHMMSS/` 结构化备份当前本地状态
3. **逐条评估**：对每条新 commit 标注冲突等级
   - 🟢 零风险（只动训练脚本、文档、非冲突文件） → 直接合入
   - 🟡 需合并（动 `voxcpm.py` / `voxcpm2.py` / `cli.py`） → 人工 cherry-pick，保留本地 +15 行类改动
   - 🔴 高冲突（动 `app.py` / `lora_ft_webui.py`） → 只手动 cherry-pick 具体 bug 修复，**绝不整体覆盖**
4. **回归验证**：启动 `app.py` 和 `lora_ft_webui.py`，确认 LoRA 热加载、超分、降噪、VAE 流式解码均正常
5. **更新本文件**：追加新一次的扫描结果与合并清单

---

## 📜 变更历史

| 日期 | 事件 |
|---|---|
| 2026-04-29 | 升级前结构化备份（`backup_before_upgrade_20260429_154548/`），基线为 `86bff0f` 之前 |
| 2026-04-29 ~ 2026-05-06 间 | 完成一次完整拉取合并，追平到 `19b6bf7`（2026-04-28） |
| 2026-05-06 | 创建本文件，记录同步状态 |
