# VoxCPM2 本地自定义改动记录

> **用途**：本项目基于开源仓库 [openbmb/VoxCPM2](https://github.com/openbmb/VoxCPM2) 搭建。  
> 以下记录了我们在原始代码基础上所做的所有改动，方便在上游更新时快速定位并重新应用。

---

## 1. `app.py` — 已恢复为上游原始行为

**历史改动（已撤销）**：曾添加长文本分句生成逻辑（`_split_sentences` + 逐句拼接），
但发现分句会导致句间韵律断裂、质量下降，已于 2026-04-16 撤销，恢复整段文本一次性生成。

---

## 2. `启动VoxCPM2.bat` — Windows 一键启动脚本

**新增文件**，非原始仓库内容。

功能：
- `chcp 65001` 设置 UTF-8 编码
- `start "" http://localhost:8808` 自动打开浏览器
- 调用 Python 3.10 运行 `app.py --port 8808`

⚠️ Python 路径硬编码为：
```
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe
```
如更换环境需修改此路径。

---

## 3. 环境修复：卸载 `torchcodec`

**问题**：开启"参考音频降噪增强"时，`torchcodec` 包（无任何包依赖它）尝试加载缺失的 FFmpeg DLL 导致崩溃。

**修复**：
```powershell
pip uninstall torchcodec -y
```

**注意**：如果重新 `pip install` 依赖时 `torchcodec` 被重新装回来，需再次卸载。

---

## 快速检查清单

当上游仓库更新后，按以下步骤操作：

1. **`app.py`**：已恢复上游原始行为，无需额外合并
2. **`启动VoxCPM2.bat`**：无需变动（独立文件）
3. **`torchcodec`**：检查是否被重新安装 → `pip show torchcodec`，如在则卸载
