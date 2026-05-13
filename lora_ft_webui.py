import os
import sys
import json
import yaml
import datetime
import subprocess
import threading
import gradio as gr
import torch
from pathlib import Path
from typing import Optional

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Default pretrained model path: prefer VoxCPM2 if it exists, fallback to VoxCPM1.5
_v2_path = project_root / "models" / "openbmb__VoxCPM2"
_v15_path = project_root / "models" / "openbmb__VoxCPM1.5"
default_pretrained_path = str(_v2_path if _v2_path.exists() else _v15_path)

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
import numpy as np
from funasr import AutoModel

# --- Localization ---
LANG_DICT = {
    "en": {
        "title": "VoxCPM LoRA WebUI",
        "tab_train": "Training",
        "tab_infer": "Inference",
        "pretrained_path": "Pretrained Model Path",
        "train_manifest": "Train Manifest (jsonl)",
        "val_manifest": "Validation Manifest (Optional)",
        "lr": "Learning Rate",
        "max_iters": "Max Iterations",
        "batch_size": "Batch Size",
        "lora_rank": "LoRA Rank",
        "lora_alpha": "LoRA Alpha",
        "save_interval": "Save Interval",
        "start_train": "Start Training",
        "stop_train": "Stop Training",
        "train_logs": "Training Logs",
        "text_to_synth": "Text to Synthesize",
        "voice_cloning": "### Voice Cloning (Optional)",
        "ref_audio": "Reference Audio",
        "ref_text": "Reference Text (Optional)",
        "select_lora": "Select LoRA Checkpoint",
        "cfg_scale": "CFG Scale",
        "infer_steps": "Inference Steps",
        "seed": "Seed",
        "gen_audio": "Generate Audio",
        "gen_output": "Generated Audio",
        "status": "Status",
        "lang_select": "Language / 语言",
        "refresh": "Refresh",
        "output_name": "Output Name (Optional, resume if exists)",
    },
    "zh": {
        "title": "VoxCPM LoRA WebUI",
        "tab_train": "训练 (Training)",
        "tab_infer": "推理 (Inference)",
        "pretrained_path": "预训练模型路径",
        "train_manifest": "训练数据清单 (jsonl)",
        "val_manifest": "验证数据清单 (可选)",
        "lr": "学习率 (Learning Rate)",
        "max_iters": "最大迭代次数",
        "batch_size": "批次大小 (Batch Size)",
        "lora_rank": "LoRA Rank",
        "lora_alpha": "LoRA Alpha",
        "save_interval": "保存间隔 (Steps)",
        "start_train": "开始训练",
        "stop_train": "停止训练",
        "train_logs": "训练日志",
        "text_to_synth": "合成文本",
        "voice_cloning": "### 声音克隆 (可选)",
        "ref_audio": "参考音频",
        "ref_text": "参考文本 (可选)",
        "select_lora": "选择 LoRA 模型",
        "cfg_scale": "CFG Scale (引导系数)",
        "infer_steps": "推理步数",
        "seed": "随机种子 (Seed)",
        "gen_audio": "生成音频",
        "gen_output": "生成结果",
        "status": "状态",
        "lang_select": "Language / 语言",
        "refresh": "刷新",
        "output_name": "输出目录名称 (可选，若存在则继续训练)",
    },
}

# Global variables
current_model: Optional[VoxCPM] = None
asr_model: Optional[AutoModel] = None
training_process: Optional[subprocess.Popen] = None
training_log = ""


def get_timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def detect_sample_rate(pretrained_path: str) -> Optional[int]:
    """Read audio_vae_config.sample_rate from the model's config.json.

    This is the AudioVAE *encoder* input rate, which is the correct rate for
    resampling training data.  Returns None when detection fails.
    """
    config_file = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(config_file):
        return None
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["audio_vae_config"]["sample_rate"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Warning: failed to detect sample_rate from {config_file}: {e}", file=sys.stderr)
        return None


def get_or_load_asr_model():
    global asr_model
    if asr_model is None:
        print("Loading ASR model (SenseVoiceSmall)...", file=sys.stderr)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            log_level="ERROR",
            device=device,
        )
    return asr_model


def recognize_audio(audio_path):
    if not audio_path:
        return ""
    try:
        model = get_or_load_asr_model()
        res = model.generate(input=audio_path, language="auto", use_itn=True)
        text = res[0]["text"].split("|>")[-1]
        return text
    except Exception as e:
        print(f"ASR Error: {e}", file=sys.stderr)
        return ""


def scan_lora_checkpoints(root_dir="lora", with_info=False):
    """
    Scans for LoRA checkpoints in the lora directory.

    Args:
        root_dir: Directory to scan for LoRA checkpoints
        with_info: If True, returns list of (path, base_model) tuples

    Returns:
        List of checkpoint paths, or list of (path, base_model) tuples if with_info=True
    """
    checkpoints = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    # Look for lora_weights.safetensors recursively
    for root, dirs, files in os.walk(root_dir):
        if "lora_weights.safetensors" in files:
            # Use the relative path from root_dir as the ID
            rel_path = os.path.relpath(root, root_dir)

            if with_info:
                # Try to read base_model from lora_config.json
                base_model = None
                lora_config_file = os.path.join(root, "lora_config.json")
                if os.path.exists(lora_config_file):
                    try:
                        with open(lora_config_file, "r", encoding="utf-8") as f:
                            lora_info = json.load(f)
                        base_model = lora_info.get("base_model", "Unknown")
                    except (json.JSONDecodeError, OSError):
                        pass
                checkpoints.append((rel_path, base_model))
            else:
                checkpoints.append(rel_path)

    # Also check for checkpoints in the default location if they exist
    default_ckpt = "checkpoints/finetune_lora"
    if os.path.exists(os.path.join(root_dir, default_ckpt)):
        # This might be covered by the walk, but good to be sure
        pass

    return sorted(checkpoints, reverse=True)


def load_lora_config_from_checkpoint(lora_path):
    """Load LoRA config from lora_config.json if available."""
    lora_config_file = os.path.join(lora_path, "lora_config.json")
    if os.path.exists(lora_config_file):
        try:
            with open(lora_config_file, "r", encoding="utf-8") as f:
                lora_info = json.load(f)
            lora_cfg_dict = lora_info.get("lora_config", {})
            if lora_cfg_dict:
                return LoRAConfig(**lora_cfg_dict), lora_info.get("base_model")
        except Exception as e:
            print(f"Warning: Failed to load lora_config.json: {e}", file=sys.stderr)
    return None, None


def get_default_lora_config():
    """Return default LoRA config for hot-swapping support."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def load_model(pretrained_path, lora_path=None):
    global current_model
    print(f"Loading model from {pretrained_path}...", file=sys.stderr)

    lora_config = None
    lora_weights_path = None

    if lora_path:
        full_lora_path = os.path.join("lora", lora_path)
        if os.path.exists(full_lora_path):
            lora_weights_path = full_lora_path
            # Try to load LoRA config from lora_config.json
            lora_config, _ = load_lora_config_from_checkpoint(full_lora_path)
            if lora_config:
                print(f"Loaded LoRA config from {full_lora_path}/lora_config.json", file=sys.stderr)
            else:
                # Fallback to default config for old checkpoints
                lora_config = get_default_lora_config()
                print("Using default LoRA config (lora_config.json not found)", file=sys.stderr)

    # Always init with a default LoRA config to allow hot-swapping later
    if lora_config is None:
        lora_config = get_default_lora_config()

    current_model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=False,
        lora_config=lora_config,
        lora_weights_path=lora_weights_path,
    )
    return "Model loaded successfully!"


def _split_sentences(text: str):
    """按中英文句末标点切分长文本为句子列表（保留标点）。
    切分符：。！？；.!?;? 以及换行。相邻短句会尽量保留；空段丢弃。
    """
    import re
    if not text:
        return []
    # 用捕获组保留分隔符，分隔符与前句合并
    parts = re.split(r"([。！？；.!?;\n]+)", text)
    out = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if re.search(r"[。！？；.!?;\n]$", p):
            s = buf.strip()
            if s:
                out.append(s)
            buf = ""
    tail = buf.strip()
    if tail:
        out.append(tail)
    return out if out else [text]


def run_inference(text, prompt_wav, prompt_text, use_prompt_text, control_instruction,
                  lora_selection, cfg_scale, steps, seed,
                  normalize, retry_badcase, seg_enabled, seg_lookback,
                  pretrained_path=None):
    """Generate audio with LoRA support.

    三种推理模式（与主 WebUI 对齐）：
      - 极致克隆：参考音频 + 参考文本（勾选 use_prompt_text）→ prompt_wav_path + prompt_text
      - 可控克隆：仅参考音频（不勾选）→ reference_wav_path（仅 V2 模型支持）
      - 风格设计：无参考音频 + control_instruction 前缀

    长文本处理（方案 A + C + D，全部假设 V2 基座）：
      - normalize: 文本规范化开关（官方默认 False）
      - retry_badcase: badcase 重试开关（官方默认 True）
      - seg_enabled: 是否按句子分段生成，避免长文本后半段漂移
      - seg_lookback: 分段时回看前 N 句干净特征（0~5），0 = 仅切分不回看
    """
    # 如果选择了 LoRA 模型且当前模型未加载，尝试从 LoRA config 读取 base_model
    if current_model is None:
        # 优先使用用户指定的预训练模型路径
        base_model_path = pretrained_path if pretrained_path and pretrained_path.strip() else default_pretrained_path

        # 如果选择了 LoRA，尝试从其 config 读取 base_model
        if lora_selection and lora_selection != "None":
            full_lora_path = os.path.join("lora", lora_selection)
            lora_config_file = os.path.join(full_lora_path, "lora_config.json")

            if os.path.exists(lora_config_file):
                try:
                    with open(lora_config_file, "r", encoding="utf-8") as f:
                        lora_info = json.load(f)
                    saved_base_model = lora_info.get("base_model")

                    if saved_base_model:
                        # 优先使用保存的 base_model 路径
                        if os.path.exists(saved_base_model):
                            base_model_path = saved_base_model
                            print(f"Using base model from LoRA config: {base_model_path}", file=sys.stderr)
                        else:
                            print(f"Warning: Saved base_model path not found: {saved_base_model}", file=sys.stderr)
                            print(f"Falling back to default: {base_model_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Failed to read base_model from LoRA config: {e}", file=sys.stderr)

        # 加载模型
        lora_to_load = lora_selection if lora_selection and lora_selection != "None" else None
        try:
            print(f"Loading base model: {base_model_path}", file=sys.stderr)
            load_model(base_model_path, lora_to_load)
            if lora_to_load:
                print(f"Model loaded with LoRA: {lora_selection}", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to load model from {base_model_path}: {str(e)}"
            print(error_msg, file=sys.stderr)
            return None, error_msg
        lora_just_loaded = lora_to_load
    else:
        lora_just_loaded = None

    # Handle LoRA hot-swapping
    assert current_model is not None, "Model must be loaded before inference"
    if lora_selection and lora_selection != "None":
        full_lora_path = os.path.join("lora", lora_selection)

        if lora_just_loaded != lora_selection:
            new_lora_config, new_base_model = load_lora_config_from_checkpoint(full_lora_path)
            current_r = current_model.tts_model.lora_config.r if current_model.tts_model.lora_config else None
            new_r = new_lora_config.r if new_lora_config else None

            if new_r is not None and current_r is not None and new_r != current_r:
                print(f"LoRA rank mismatch (model r={current_r}, checkpoint r={new_r}), reloading...", file=sys.stderr)
                reload_base = (
                    new_base_model if new_base_model and os.path.exists(new_base_model)
                    else (pretrained_path if pretrained_path and pretrained_path.strip() else default_pretrained_path)
                )
                try:
                    load_model(reload_base, lora_selection)
                except Exception as e:
                    return None, f"Failed to reload model for LoRA rank change: {e}"
            else:
                print(f"Hot-loading LoRA: {full_lora_path}", file=sys.stderr)
                try:
                    current_model.load_lora(full_lora_path)
                except Exception as e:
                    print(f"Error loading LoRA: {e}", file=sys.stderr)
                    return None, f"Error loading LoRA: {e}"
        current_model.set_lora_enabled(True)
    else:
        print("Disabling LoRA", file=sys.stderr)
        current_model.set_lora_enabled(False)

    # === 随机种子处理：无论用户是否指定，都计算一个 effective_seed 并显式设定 ===
    import random as _random
    if seed is None or int(seed) == -1:
        effective_seed = _random.randint(0, 2**31 - 1)
    else:
        effective_seed = int(seed)
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)

    # === 三种模式参数解析 ===
    audio_path = prompt_wav if (prompt_wav and prompt_wav.strip()) else None
    # use_prompt_text 决定是否启用 prompt_text 分支（极致克隆）
    prompt_text_clean = (prompt_text or "").strip() if use_prompt_text else None
    prompt_text_clean = prompt_text_clean if prompt_text_clean else None
    # control_instruction 仅在非极致克隆模式下生效（可控克隆 / 风格设计）
    control = (control_instruction or "").strip() if not use_prompt_text else ""

    # 若勾选极致克隆但缺失参考文本，尝试 ASR 自动识别
    if use_prompt_text and audio_path and not prompt_text_clean:
        print("参考音频已提供但缺少文本，自动识别中...", file=sys.stderr)
        try:
            prompt_text_clean = recognize_audio(audio_path)
            if prompt_text_clean:
                print(f"自动识别文本: {prompt_text_clean}", file=sys.stderr)
            else:
                return None, "错误：无法识别参考音频内容，请手动填写参考文本"
        except Exception as e:
            return None, f"错误：自动识别参考音频失败 - {str(e)}"

    # 判定模式
    if audio_path and prompt_text_clean:
        mode = "极致克隆 (prompt_wav + prompt_text)"
    elif audio_path:
        mode = "可控克隆 (reference_wav only)"
    else:
        mode = f"风格设计 (control: {control[:40] if control else 'zero-shot'})"
    print(f"[Inference] Mode: {mode}", file=sys.stderr)

    # 构建最终文本（control_instruction 作为前缀）
    final_text = f"({control}){text}" if control else text

    # 按模式选择正确的官方 API 参数通道：
    #   极致克隆：prompt_wav_path + prompt_text（成对，续写式克隆）
    #   可控克隆：reference_wav_path（仅 V2，音色隔离通道）
    #   风格设计：两者都不传
    gen_kwargs = dict(
        text=final_text,
        cfg_value=cfg_scale,
        inference_timesteps=steps,
        denoise=False,
        normalize=bool(normalize),
        retry_badcase=bool(retry_badcase),
    )
    if audio_path and prompt_text_clean:
        gen_kwargs["prompt_wav_path"] = audio_path
        gen_kwargs["prompt_text"] = prompt_text_clean
    elif audio_path:
        # 可控克隆仅 V2 模型支持（无兜底，让官方报错冒出来足够明确）
        gen_kwargs["reference_wav_path"] = audio_path

    try:
        # ================= 分段生成路径 =================
        if seg_enabled:
            from voxcpm.core import next_and_close

            sr = current_model.tts_model.sample_rate
            tts = current_model.tts_model

            # 1) 一次性构建 prompt_cache（按当前模式对应的通道）
            if audio_path and prompt_text_clean:
                fixed_cache = tts.build_prompt_cache(
                    prompt_text=prompt_text_clean,
                    prompt_wav_path=audio_path,
                )
            elif audio_path:
                # 可控克隆 → reference_wav_path
                fixed_cache = tts.build_prompt_cache(
                    prompt_text=None,
                    prompt_wav_path=None,
                    reference_wav_path=audio_path,
                )
            else:
                fixed_cache = None  # 零样本风格设计

            # 2) 切句 + 可选 text_normalize
            raw_sentences = _split_sentences(text)
            sentences = []
            for s in raw_sentences:
                if normalize and hasattr(current_model, "text_normalizer") and current_model.text_normalizer:
                    try:
                        s = current_model.text_normalizer.normalize(s)
                    except Exception:
                        pass
                if s and s.strip():
                    sentences.append(s.strip())
            if not sentences:
                return None, "分段失败：文本切分后为空"

            N = max(0, min(5, int(seg_lookback)))
            print(
                f"[Seg] 分段生成启动 | {len(sentences)} 段 | lookback={N} | control={'有' if control else '空'}",
                file=sys.stderr,
            )

            def _gen_one(text_str, cache):
                gen = tts._generate_with_prompt_cache(
                    target_text=text_str,
                    prompt_cache=cache if cache is not None else tts.build_prompt_cache(
                        prompt_text=None, prompt_wav_path=None
                    ),
                    inference_timesteps=int(steps),
                    cfg_value=float(cfg_scale),
                    streaming=False,
                )
                wav_t, _, feat = next_and_close(gen)
                return wav_t.squeeze(0).cpu().numpy(), feat

            wav_segments = []
            prev_feats = []  # 队列：最近 N 句的 (带 control 的文本, clean_feat)

            for i, s in enumerate(sentences):
                final_s = f"({control}){s}" if control else s
                print(f"  [{i+1}/{len(sentences)}] {final_s[:60]}", file=sys.stderr)

                first_or_no_lookback = (i == 0) or (N == 0) or (not prev_feats)

                if first_or_no_lookback:
                    # 首句 / 禁用回看：直接用原 cache
                    wav_np, clean_feat = _gen_one(final_s, fixed_cache)
                elif not control:
                    # control 为空 → Pass 1 与 Pass 2 文本相同 → 单 Pass 退化
                    lookback_cache = fixed_cache
                    for prev_text, prev_feat in prev_feats[-N:]:
                        lookback_cache = tts.merge_prompt_cache(
                            lookback_cache, new_text=prev_text, new_audio_feat=prev_feat
                        )
                    wav_np, clean_feat = _gen_one(s, lookback_cache)
                else:
                    # 双 Pass（control 非空）
                    # Pass 1：带 control 在原 cache 上生成 → 仅取 clean_feat
                    _, clean_feat = _gen_one(final_s, fixed_cache)
                    # Pass 2：lookback cache + 纯句子 → 避免念 control
                    lookback_cache = fixed_cache
                    for prev_text, prev_feat in prev_feats[-N:]:
                        lookback_cache = tts.merge_prompt_cache(
                            lookback_cache, new_text=prev_text, new_audio_feat=prev_feat
                        )
                    wav_np, _ = _gen_one(s, lookback_cache)

                wav_segments.append(wav_np)
                if N > 0:
                    prev_feats.append((final_s, clean_feat))
                    if len(prev_feats) > N:
                        prev_feats.pop(0)

            final_wav = np.concatenate(wav_segments) if len(wav_segments) > 1 else wav_segments[0]
            dur = len(final_wav) / sr
            return (sr, final_wav), f"分段生成成功 | {len(sentences)} 段 / {dur:.1f}s | lookback={N} | Seed: {effective_seed} | 模式: {mode}"

        # ================= 常规路径 =================
        audio_np = current_model.generate(**gen_kwargs)
        return (current_model.tts_model.sample_rate, audio_np), f"生成成功 | Seed: {effective_seed} | 模式: {mode}"
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error (Seed: {effective_seed}): {str(e)}"


def start_training(
    pretrained_path,
    train_manifest,
    val_manifest,
    learning_rate,
    num_iters,
    batch_size,
    lora_rank,
    lora_alpha,
    save_interval,
    output_name="",
    # Advanced options
    grad_accum_steps=1,
    num_workers=2,
    log_interval=10,
    valid_interval=1000,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=None,
    sample_rate=44100,
    max_grad_norm=1.0,
    # LoRA advanced
    enable_lm=True,
    enable_dit=True,
    enable_proj=False,
    dropout=0.0,
    tensorboard_path="",
    # Distribution options
    hf_model_id="",
    distribute=False,
):
    global training_log

    if training_process is not None and training_process.poll() is None:
        return "Training is already running!"

    if output_name and output_name.strip():
        timestamp = output_name.strip()
    else:
        timestamp = get_timestamp_str()

    save_dir = os.path.join("lora", timestamp)
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    logs_dir = os.path.join(save_dir, "logs")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Auto-detect sample_rate from model config.json to prevent mismatch
    detected_sr = detect_sample_rate(pretrained_path)
    if detected_sr is not None:
        if int(sample_rate) != detected_sr:
            training_log += (
                f"[Auto-fix] sample_rate changed from {int(sample_rate)} to {detected_sr} "
                f"(read from {pretrained_path}/config.json audio_vae_config.sample_rate)\n"
            )
        sample_rate = detected_sr

    # Create config dictionary
    # Resolve max_steps default
    resolved_max_steps = int(max_steps) if max_steps not in (None, "", 0) else int(num_iters)

    # Auto-detect out_sample_rate from model config
    out_sample_rate = 0
    config_file = os.path.join(pretrained_path, "config.json")
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            out_sr = cfg.get("audio_vae_config", {}).get("out_sample_rate")
            if out_sr:
                out_sample_rate = int(out_sr)
        except Exception:
            pass

    config = {
        "pretrained_path": pretrained_path,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "sample_rate": int(sample_rate),
        "out_sample_rate": out_sample_rate,
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "num_workers": int(num_workers),
        "num_iters": int(num_iters),
        "log_interval": int(log_interval),
        "valid_interval": int(valid_interval),
        "save_interval": int(save_interval),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_steps": int(warmup_steps),
        "max_steps": resolved_max_steps,
        "max_grad_norm": float(max_grad_norm),
        "save_path": checkpoints_dir,
        "tensorboard": tensorboard_path if tensorboard_path else logs_dir,
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": bool(enable_lm),
            "enable_dit": bool(enable_dit),
            "enable_proj": bool(enable_proj),
            "r": int(lora_rank),
            "alpha": int(lora_alpha),
            "dropout": float(dropout),
            "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
    }

    # Add distribution options if provided
    if hf_model_id and hf_model_id.strip():
        config["hf_model_id"] = hf_model_id.strip()
    if distribute:
        config["distribute"] = True

    config_path = os.path.join(save_dir, "train_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    cmd = [sys.executable, "scripts/train_voxcpm_finetune.py", "--config_path", config_path]

    training_log = f"Starting training...\nConfig saved to {config_path}\nOutput dir: {save_dir}\n"

    def run_process():
        global training_process, training_log
        training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        assert training_process.stdout is not None
        for line in training_process.stdout:
            training_log += line
            # Keep log size manageable
            if len(training_log) > 100000:
                training_log = training_log[-100000:]

        training_process.wait()
        training_log += f"\nTraining finished with code {training_process.returncode}"

    threading.Thread(target=run_process, daemon=True).start()

    return f"Training started! Check 'lora/{timestamp}'"


def get_training_log():
    return training_log


def stop_training():
    global training_log
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        training_log += "\nTraining terminated by user."
        return "Training stopped."
    return "No training running."


# --- GUI Layout ---

# 自定义CSS样式
custom_css = """
/* 整体主题样式 */
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* 标题区域样式 - 扁平化设计 */
.title-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    padding: 15px 25px;
    margin-bottom: 15px;
    border: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.title-section h1 {
    color: white;
    text-shadow: none;
    font-weight: 600;
    margin: 0;
    font-size: 28px;
    line-height: 1.2;
}

.title-section h3 {
    color: rgba(255, 255, 255, 0.9);
    font-weight: 400;
    margin-top: 5px;
    font-size: 14px;
    line-height: 1.3;
}

.title-section p {
    color: rgba(255, 255, 255, 0.85);
    font-size: 13px;
    margin: 5px 0 0 0;
    line-height: 1.3;
}

/* 标签页样式 */
.tabs {
    background: white;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* 按钮样式增强 */
.button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-weight: 600;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
}

.button-stop {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-weight: 600;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
}

.button-stop:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(250, 112, 154, 0.4);
}

.button-refresh {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 500;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 2px 10px rgba(132, 250, 176, 0.3);
}

.button-refresh:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(132, 250, 176, 0.4);
}

/* 表单区域样式 */
.form-section {
    background: white;
    border-radius: 20px;
    padding: 30px;
    margin: 15px 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
}

/* 输入框样式 */
.input-field {
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    padding: 12px 16px;
    transition: all 0.3s ease;
    background: #fafafa;
}

.input-field:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    background: white;
}

/* 滑块样式 */
.slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    outline: none;
    opacity: 0.8;
    transition: opacity 0.2s;
}

.slider:hover {
    opacity: 1;
}

.slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
    cursor: pointer;
    border: 3px solid #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

.slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
    cursor: pointer;
    border: 3px solid #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

/* 折叠面板样式 */
.accordion {
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    overflow: hidden;
    background: white;
}

.accordion-header {
    background: linear-gradient(135deg, #f5f7fa 0%, #e3e7ed 100%);
    padding: 15px 20px;
    font-weight: 600;
    color: #333;
}

/* 状态显示样式 */
.status-success {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(132, 250, 176, 0.3);
}

.status-error {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
}

/* 语言切换按钮样式 - 扁平化 */
.lang-selector {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border-radius: 8px;
    padding: 8px 12px;
    border: 1px solid rgba(255, 255, 255, 0.4);
}

.lang-selector label.gr-box {
    color: white !important;
    font-weight: 600;
    margin-bottom: 8px !important;
}

/* 单选按钮组样式 */
.lang-selector fieldset,
.lang-selector .gr-form {
    gap: 10px !important;
    display: flex !important;
}

/* 单选按钮容器 - 扁平化 (未选中状态 - 较浅的深色) */
.lang-selector label.gr-radio-label {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6)) !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    color: white !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    cursor: pointer !important;
    margin: 0 4px !important;
}

/* 选中的单选按钮 - 扁平化 (更深的深色背景) */
.lang-selector input[type="radio"]:checked + label,
.lang-selector label.gr-radio-label:has(input:checked) {
    background: linear-gradient(135deg, #5568d3, #6b4c9a) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2) !important;
    transform: none !important;
}

/* 未选中的单选按钮悬停效果 - 扁平化 */
.lang-selector label.gr-radio-label:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.75), rgba(118, 75, 162, 0.75)) !important;
    border-color: rgba(255, 255, 255, 0.7) !important;
    transform: none !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
}

/* 隐藏原始的单选按钮圆点 */
.lang-selector input[type="radio"] {
    opacity: 0;
    position: absolute;
}

/* Gradio Radio 特定样式 - 扁平化 */
.lang-selector .wrap {
    gap: 8px !important;
}

.lang-selector .wrap > label {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6)) !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    color: white !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.lang-selector .wrap > label.selected {
    background: linear-gradient(135deg, #5568d3, #6b4c9a) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2) !important;
}

/* 标签样式优化 */
label {
    color: #333;
    font-weight: 500;
    margin-bottom: 8px;
}

/* Markdown 标题样式 */
.markdown-text h4 {
    color: #667eea;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 10px;
}

/* 参数组件间距优化 */
.form-section > div {
    margin-bottom: 15px;
}

/* Slider 组件样式优化 */
.gr-slider {
    padding: 10px 0;
}

/* Number 输入框优化 */
.gr-number {
    max-width: 100%;
}

/* 按钮容器优化 */
.gr-button {
    min-height: 45px;
    font-size: 16px;
}

/* 三栏布局优化 */
#component-0 .gr-row {
    gap: 20px;
}

/* 生成按钮特殊样式 */
.button-primary.gr-button-lg {
    min-height: 55px;
    font-size: 18px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* 刷新按钮小尺寸 */
.button-refresh.gr-button-sm {
    min-height: 38px;
    font-size: 14px;
    margin-top: 5px;
    margin-bottom: 15px;
}

/* 信息提示文字样式 */
.gr-info {
    font-size: 13px;
    color: #666;
    margin-top: 5px;
}

/* 区域标题样式优化 */
.form-section h4 {
    color: #667eea;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f0f0f0;
}

.form-section strong {
    color: #667eea;
    font-size: 15px;
    display: block;
    margin: 15px 0 10px 0;
}
"""

with gr.Blocks(title="VoxCPM LoRA WebUI", theme=gr.themes.Soft(), css=custom_css) as app:

    # State for language
    lang_state = gr.State("zh")  # Default to Chinese

    # 标题区域
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=3):
            title_md = gr.Markdown("""
            # VoxCPM LoRA WebUI
            ### 强大的语音合成和 LoRA 微调工具

            支持语音克隆、LoRA 模型训练和推理的完整解决方案
            """)
        with gr.Column(scale=1):
            lang_btn = gr.Radio(
                choices=["en", "zh"], value="zh", label="Language / 语言", elem_classes="lang-selector"
            )

    with gr.Tabs(elem_classes="tabs") as tabs:
        # === Training Tab ===
        with gr.Tab("训练 (Training)") as tab_train:
            gr.Markdown("""
            ### 模型训练设置
            配置你的 LoRA 微调训练参数
            """)

            with gr.Row():
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 基础配置")

                    train_pretrained_path = gr.Textbox(
                        label="预训练模型路径", value=default_pretrained_path, elem_classes="input-field"
                    )
                    train_manifest = gr.Textbox(
                        label="训练数据清单 (jsonl)",
                        value="examples/train_data_example.jsonl",
                        elem_classes="input-field",
                    )
                    val_manifest = gr.Textbox(label="验证数据清单 (可选)", value="", elem_classes="input-field")

                    gr.Markdown("#### 训练参数")

                    with gr.Row():
                        lr = gr.Number(label="学习率 (Learning Rate)", value=1e-4, elem_classes="input-field")
                        num_iters = gr.Number(
                            label="最大迭代次数", value=2000, precision=0, elem_classes="input-field"
                        )
                        batch_size = gr.Number(
                            label="批次大小 (Batch Size)", value=1, precision=0, elem_classes="input-field"
                        )

                    with gr.Row():
                        lora_rank = gr.Number(label="LoRA Rank", value=32, precision=0, elem_classes="input-field")
                        lora_alpha = gr.Number(label="LoRA Alpha", value=16, precision=0, elem_classes="input-field")
                        save_interval = gr.Number(
                            label="保存间隔 (Steps)", value=1000, precision=0, elem_classes="input-field"
                        )

                    output_name = gr.Textbox(
                        label="输出目录名称 (可选，若存在则继续训练)", value="", elem_classes="input-field"
                    )

                    with gr.Row():
                        start_btn = gr.Button("开始训练", variant="primary", elem_classes="button-primary")
                        stop_btn = gr.Button("停止训练", variant="stop", elem_classes="button-stop")

                    with gr.Accordion("高级选项 (Advanced)", open=False, elem_classes="accordion"):
                        with gr.Row():
                            grad_accum_steps = gr.Number(label="梯度累积 (grad_accum_steps)", value=1, precision=0)
                            num_workers = gr.Number(label="数据加载线程 (num_workers)", value=2, precision=0)
                            log_interval = gr.Number(label="日志间隔 (log_interval)", value=10, precision=0)
                        with gr.Row():
                            valid_interval = gr.Number(label="验证间隔 (valid_interval)", value=1000, precision=0)
                            weight_decay = gr.Number(label="权重衰减 (weight_decay)", value=0.01)
                            warmup_steps = gr.Number(label="warmup_steps", value=100, precision=0)
                        with gr.Row():
                            max_steps = gr.Number(label="最大步数 (max_steps, 0→默认num_iters)", value=0, precision=0)
                            sample_rate = gr.Number(label="采样率 (sample_rate)", value=44100, precision=0)
                            max_grad_norm = gr.Number(label="梯度裁剪 (max_grad_norm, 0=关闭)", value=1.0)
                        with gr.Row():
                            tensorboard_path = gr.Textbox(label="Tensorboard 路径 (可选)", value="")
                            enable_lm = gr.Checkbox(label="启用 LoRA LM (enable_lm)", value=True)
                            enable_dit = gr.Checkbox(label="启用 LoRA DIT (enable_dit)", value=True)
                        with gr.Row():
                            enable_proj = gr.Checkbox(label="启用投影 (enable_proj)", value=False)
                            dropout = gr.Number(label="LoRA Dropout", value=0.0)

                        gr.Markdown("#### 分发选项 (Distribution)")
                        with gr.Row():
                            hf_model_id = gr.Textbox(
                                label="HuggingFace Model ID (e.g., openbmb/VoxCPM2)", value=""
                            )
                            distribute = gr.Checkbox(label="分发模式 (distribute)", value=False)

                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 训练日志")
                    logs_out = gr.TextArea(
                        label="",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        elem_classes="input-field",
                        show_label=False,
                    )

            def on_pretrained_path_change(path):
                """Auto-detect sample_rate when pretrained model path changes."""
                sr = detect_sample_rate(path)
                if sr is not None:
                    return gr.update(value=sr)
                return gr.update()

            train_pretrained_path.change(
                on_pretrained_path_change,
                inputs=[train_pretrained_path],
                outputs=[sample_rate],
            )

            start_btn.click(
                start_training,
                inputs=[
                    train_pretrained_path,
                    train_manifest,
                    val_manifest,
                    lr,
                    num_iters,
                    batch_size,
                    lora_rank,
                    lora_alpha,
                    save_interval,
                    output_name,
                    # advanced
                    grad_accum_steps,
                    num_workers,
                    log_interval,
                    valid_interval,
                    weight_decay,
                    warmup_steps,
                    max_steps,
                    sample_rate,
                    max_grad_norm,
                    enable_lm,
                    enable_dit,
                    enable_proj,
                    dropout,
                    tensorboard_path,
                    # distribution
                    hf_model_id,
                    distribute,
                ],
                outputs=[logs_out],  # Initial message
            )
            stop_btn.click(stop_training, outputs=[logs_out])

            # Log refresher
            timer = gr.Timer(1)
            timer.tick(get_training_log, outputs=logs_out)

        # === Inference Tab ===
        with gr.Tab("推理 (Inference)") as tab_infer:
            gr.Markdown("""
            ### 语音合成
            使用训练好的 LoRA 模型生成语音，支持 LoRA 微调和声音克隆
            """)

            with gr.Row():
                # 左栏：输入配置 (35%)
                with gr.Column(scale=35, elem_classes="form-section"):
                    gr.Markdown("#### 输入配置")

                    infer_text = gr.TextArea(
                        label="合成文本",
                        value="Hello, this is a test of the VoxCPM LoRA model.",
                        elem_classes="input-field",
                        lines=4,
                        placeholder="输入要合成的文本内容...",
                    )

                    gr.Markdown("**声音克隆（可选）**")

                    prompt_wav = gr.Audio(label="参考音频", type="filepath", elem_classes="input-field")

                    prompt_text = gr.Textbox(
                        label="参考文本（可选）",
                        elem_classes="input-field",
                        placeholder="如不填写且勾选极致克隆，将自动识别参考音频内容",
                    )

                    use_prompt_text = gr.Checkbox(
                        label="启用极致克隆（使用参考文本）",
                        value=True,
                        info="勾选：极致克隆（需参考音频+参考文本） | 取消：可控克隆（仅音色复刻）或风格设计",
                    )

                    control_instruction = gr.Textbox(
                        label="风格/控制指令（可选）",
                        elem_classes="input-field",
                        placeholder="取消极致克隆后生效，如：'用温柔的语气说' 或 '悲伤地'",
                        lines=2,
                    )

                    with gr.Accordion("高级选项（长文本/质量控制）", open=False):
                        normalize_cb = gr.Checkbox(
                            label="启用文本规范化 (normalize)",
                            value=False,
                            info="对标点/数字做标准化。官方默认 False",
                        )
                        retry_badcase_cb = gr.Checkbox(
                            label="启用 badcase 重试 (retry_badcase)",
                            value=True,
                            info="检测到坏例自动重试。官方默认 True；若发现重试反而更糟可关闭",
                        )
                        seg_enabled_cb = gr.Checkbox(
                            label="启用分段生成（长文本防糊）",
                            value=False,
                            info="按句号等标点切分后逐句生成，缓解 30s+ 长文本后半段漂移/嗡嗡",
                        )
                        seg_lookback_sl = gr.Slider(
                            label="分段回看句数 (lookback)",
                            minimum=0,
                            maximum=5,
                            value=1,
                            step=1,
                            info="0=仅切分不回看；1~5=回看前 N 句干净特征维持连贯性（V2 专属）",
                        )

                # 中栏：模型选择和参数配置 (35%)
                with gr.Column(scale=35, elem_classes="form-section"):
                    gr.Markdown("#### 模型选择")

                    lora_select = gr.Dropdown(
                        label="LoRA 模型",
                        choices=["None"] + scan_lora_checkpoints(),
                        value="None",
                        interactive=True,
                        elem_classes="input-field",
                        info="选择训练好的 LoRA 模型，或选择 None 使用基础模型",
                    )

                    refresh_lora_btn = gr.Button("刷新模型列表", elem_classes="button-refresh", size="sm")

                    gr.Markdown("#### 生成参数")

                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=5.0,
                        value=2.0,
                        step=0.1,
                        info="引导系数，值越大越贴近提示",
                    )

                    steps = gr.Slider(
                        label="推理步数",
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        info="生成质量与步数成正比，但耗时更长",
                    )

                    seed = gr.Number(
                        label="随机种子",
                        value=-1,
                        precision=0,
                        elem_classes="input-field",
                        info="-1 为随机，固定值可复现结果",
                    )

                    generate_btn = gr.Button("生成音频", variant="primary", elem_classes="button-primary", size="lg")

                # 右栏：生成结果 (30%)
                with gr.Column(scale=30, elem_classes="form-section"):
                    gr.Markdown("#### 生成结果")

                    audio_out = gr.Audio(label="", elem_classes="input-field", show_label=False)

                    gr.Markdown("#### 状态信息")

                    status_out = gr.Textbox(
                        label="",
                        interactive=False,
                        elem_classes="input-field",
                        show_label=False,
                        lines=3,
                        placeholder="等待生成...",
                    )

            def refresh_loras():
                # 获取 LoRA checkpoints 及其 base model 信息
                checkpoints_with_info = scan_lora_checkpoints(with_info=True)
                choices = ["None"] + [ckpt[0] for ckpt in checkpoints_with_info]

                # 输出调试信息
                print(f"刷新 LoRA 列表: 找到 {len(checkpoints_with_info)} 个检查点", file=sys.stderr)
                for ckpt_path, base_model in checkpoints_with_info:
                    if base_model:
                        print(f"  - {ckpt_path} (Base Model: {base_model})", file=sys.stderr)
                    else:
                        print(f"  - {ckpt_path}", file=sys.stderr)

                return gr.update(choices=choices, value="None")

            refresh_lora_btn.click(refresh_loras, outputs=[lora_select])

            # Auto-recognize audio when uploaded
            prompt_wav.change(fn=recognize_audio, inputs=[prompt_wav], outputs=[prompt_text])

            generate_btn.click(
                run_inference,
                inputs=[
                    infer_text,
                    prompt_wav,
                    prompt_text,
                    use_prompt_text,
                    control_instruction,
                    lora_select,
                    cfg_scale,
                    steps,
                    seed,
                    normalize_cb,
                    retry_badcase_cb,
                    seg_enabled_cb,
                    seg_lookback_sl,
                    train_pretrained_path,
                ],
                outputs=[audio_out, status_out],
            )

    # --- Language Switching Logic ---
    def change_language(lang):
        d = LANG_DICT[lang]
        # Labels for advanced options
        if lang == "zh":
            adv = {
                "grad_accum_steps": "梯度累积 (grad_accum_steps)",
                "num_workers": "数据加载线程 (num_workers)",
                "log_interval": "日志间隔 (log_interval)",
                "valid_interval": "验证间隔 (valid_interval)",
                "weight_decay": "权重衰减 (weight_decay)",
                "warmup_steps": "warmup_steps",
                "max_steps": "最大步数 (max_steps)",
                "sample_rate": "采样率 (sample_rate)",
                "max_grad_norm": "梯度裁剪 (max_grad_norm, 0=关闭)",
                "enable_lm": "启用 LoRA LM (enable_lm)",
                "enable_dit": "启用 LoRA DIT (enable_dit)",
                "enable_proj": "启用投影 (enable_proj)",
                "dropout": "LoRA Dropout",
                "tensorboard_path": "Tensorboard 路径 (可选)",
                "hf_model_id": "HuggingFace Model ID (e.g., openbmb/VoxCPM2)",
                "distribute": "分发模式 (distribute)",
            }
        else:
            adv = {
                "grad_accum_steps": "Grad Accum Steps",
                "num_workers": "Num Workers",
                "log_interval": "Log Interval",
                "valid_interval": "Valid Interval",
                "weight_decay": "Weight Decay",
                "warmup_steps": "Warmup Steps",
                "max_steps": "Max Steps",
                "sample_rate": "Sample Rate",
                "max_grad_norm": "Max Grad Norm (0=disabled)",
                "enable_lm": "Enable LoRA LM",
                "enable_dit": "Enable LoRA DIT",
                "enable_proj": "Enable Projection",
                "dropout": "LoRA Dropout",
                "tensorboard_path": "Tensorboard Path (Optional)",
                "hf_model_id": "HuggingFace Model ID (e.g., openbmb/VoxCPM2)",
                "distribute": "Distribute Mode",
            }

        return (
            gr.update(value=f"# {d['title']}"),
            gr.update(label=d["tab_train"]),
            gr.update(label=d["tab_infer"]),
            gr.update(label=d["pretrained_path"]),
            gr.update(label=d["train_manifest"]),
            gr.update(label=d["val_manifest"]),
            gr.update(label=d["lr"]),
            gr.update(label=d["max_iters"]),
            gr.update(label=d["batch_size"]),
            gr.update(label=d["lora_rank"]),
            gr.update(label=d["lora_alpha"]),
            gr.update(label=d["save_interval"]),
            gr.update(label=d["output_name"]),
            gr.update(value=d["start_train"]),
            gr.update(value=d["stop_train"]),
            gr.update(label=d["train_logs"]),
            # Advanced options (must match outputs order)
            gr.update(label=adv["grad_accum_steps"]),
            gr.update(label=adv["num_workers"]),
            gr.update(label=adv["log_interval"]),
            gr.update(label=adv["valid_interval"]),
            gr.update(label=adv["weight_decay"]),
            gr.update(label=adv["warmup_steps"]),
            gr.update(label=adv["max_steps"]),
            gr.update(label=adv["sample_rate"]),
            gr.update(label=adv["max_grad_norm"]),
            gr.update(label=adv["tensorboard_path"]),
            gr.update(label=adv["enable_lm"]),
            gr.update(label=adv["enable_dit"]),
            gr.update(label=adv["enable_proj"]),
            gr.update(label=adv["dropout"]),
            # Distribution options
            gr.update(label=adv["hf_model_id"]),
            gr.update(label=adv["distribute"]),
            # Inference section
            gr.update(label=d["text_to_synth"]),
            gr.update(label=d["ref_audio"]),
            gr.update(label=d["ref_text"]),
            gr.update(label=d["select_lora"]),
            gr.update(value=d["refresh"]),
            gr.update(label=d["cfg_scale"]),
            gr.update(label=d["infer_steps"]),
            gr.update(label=d["seed"]),
            gr.update(value=d["gen_audio"]),
            gr.update(label=d["gen_output"]),
            gr.update(label=d["status"]),
        )

    lang_btn.change(
        change_language,
        inputs=[lang_btn],
        outputs=[
            title_md,
            tab_train,
            tab_infer,
            train_pretrained_path,
            train_manifest,
            val_manifest,
            lr,
            num_iters,
            batch_size,
            lora_rank,
            lora_alpha,
            save_interval,
            output_name,
            start_btn,
            stop_btn,
            logs_out,
            # advanced outputs
            grad_accum_steps,
            num_workers,
            log_interval,
            valid_interval,
            weight_decay,
            warmup_steps,
            max_steps,
            sample_rate,
            max_grad_norm,
            tensorboard_path,
            enable_lm,
            enable_dit,
            enable_proj,
            dropout,
            # distribution outputs
            hf_model_id,
            distribute,
            infer_text,
            prompt_wav,
            prompt_text,
            lora_select,
            refresh_lora_btn,
            cfg_scale,
            steps,
            seed,
            generate_btn,
            audio_out,
            status_out,
        ],
    )

if __name__ == "__main__":
    # Ensure lora directory exists
    os.makedirs("lora", exist_ok=True)
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
