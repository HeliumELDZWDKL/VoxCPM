import os
import re
import sys
import json
import logging
import numpy as np
import torch
import gradio as gr
from typing import Optional, Tuple, List
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.voxcpm import LoRAConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    '`Target Text`: *"I never asked you to stay… It\'s not like I care or anything. '
    'But… why does it still hurt so much now that you\'re gone?"*  \n\n'
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "audio_sr_label": "Audio Super-Resolution (AP-BWE)",
        "audio_sr_info": "Rebuild high-frequency details via bandwidth extension (48k→24k→48k). Helps reduce blurriness in long texts",
        "seg_mode_label": "Segmented Generation Mode",
        "seg_mode_none": "Disabled (whole text)",
        "seg_mode_clear": "Clear KV cache per sentence",
        "seg_mode_lookback": "Lookback 1 sentence KV cache",
        "seg_mode_info": "Controls how KV cache is handled during segmented generation to balance quality and prosody continuity",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "audio_sr_label": "音频超分辨率 (AP-BWE)",
        "audio_sr_info": "通过带宽扩展重建高频细节 (48k→24k→48k)，可改善长文本后半段音质模糊的问题",
        "seg_mode_label": "分段生成模式",
        "seg_mode_none": "不分段（整段生成）",
        "seg_mode_clear": "每句清除 KV Cache",
        "seg_mode_lookback": "回看一句 KV Cache",
        "seg_mode_info": "控制分段生成时 KV Cache 的处理方式，平衡音质与语调连贯性",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,       # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- LoRA Helpers ----------

def get_default_lora_config():
    """Return default LoRA config so the model supports hot-swapping later."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def scan_lora_checkpoints(root_dir="lora"):
    """Scan for LoRA checkpoints under *root_dir* and return sorted list of relative paths."""
    checkpoints = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    for root, _dirs, files in os.walk(root_dir):
        if "lora_weights.safetensors" in files:
            checkpoints.append(os.path.relpath(root, root_dir))
    return sorted(checkpoints, reverse=True)


def load_lora_config_from_checkpoint(lora_path):
    """Try to load LoRA config from lora_config.json inside *lora_path*."""
    cfg_file = os.path.join(lora_path, "lora_config.json")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            cfg_dict = info.get("lora_config", {})
            if cfg_dict:
                return LoRAConfig(**cfg_dict), info.get("base_model")
        except Exception as e:
            logger.warning(f"Failed to load lora_config.json: {e}")
    return None, None


# ---------- Model ----------

class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running on device: {self.device}")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id
        self._current_lora: Optional[str] = None  # currently loaded LoRA path
        self._current_lora_rank: Optional[int] = None  # rank of current LoRA config

    def get_or_load_voxcpm(self, lora_config=None, optimize=True) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        if lora_config is None:
            lora_config = get_default_lora_config()
        logger.info(f"Loading model: {self._model_id} (LoRA r={lora_config.r}, optimize={optimize})")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=optimize,
            lora_config=lora_config,
        )
        self._current_lora_rank = lora_config.r
        logger.info("Model loaded successfully (with LoRA support).")
        return self.voxcpm_model

    def _reload_model_with_config(self, lora_config) -> None:
        """Reload model with a different LoRA config (when rank changes)."""
        logger.info(f"Rank changed ({self._current_lora_rank} -> {lora_config.r}), reloading model...")
        del self.voxcpm_model
        self.voxcpm_model = None
        self._current_lora = None
        self._current_lora_rank = None
        torch.cuda.empty_cache()
        # optimize=False to avoid CUDA Graph state conflict on reload
        self.get_or_load_voxcpm(lora_config=lora_config, optimize=False)

    def swap_lora(self, lora_selection: Optional[str]) -> None:
        """Hot-swap LoRA weights. Auto-reloads model if rank changes."""
        model = self.get_or_load_voxcpm()
        if not lora_selection or lora_selection == "None":
            if self._current_lora is not None:
                logger.info("Disabling LoRA")
                model.set_lora_enabled(False)
                self._current_lora = None
            return
        full_path = os.path.join("lora", lora_selection)
        if full_path != self._current_lora:
            # Read LoRA config to check rank compatibility
            lora_cfg, _ = load_lora_config_from_checkpoint(full_path)
            if lora_cfg is None:
                lora_cfg = get_default_lora_config()
            # If rank differs, must reload entire model
            if self._current_lora_rank != lora_cfg.r:
                self._reload_model_with_config(lora_cfg)
                model = self.voxcpm_model
            logger.info(f"Loading LoRA: {full_path} (r={lora_cfg.r})")
            model.load_lora(full_path)
            self._current_lora = full_path
        model.set_lora_enabled(True)

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = False,
        inference_timesteps: int = 10,
    ) -> Tuple[int, np.ndarray]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        final_text = f"({control}){text}" if control else text
        logger.info(f"Generating audio: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
        )
        wav = current_model.generate(**generate_kwargs)
        return (current_model.tts_model.sample_rate, wav)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences by Chinese/English punctuation."""
        # Split on sentence-ending punctuation, keeping the punctuation attached
        parts = re.split(r'(?<=[。！？；\n.!?;])', text)
        sentences = [s.strip() for s in parts if s.strip()]
        if not sentences:
            sentences = [text]
        return sentences

    def generate_tts_audio_segmented(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = False,
        inference_timesteps: int = 10,
        lookback: bool = False,
    ) -> Tuple[int, np.ndarray]:
        """Generate audio sentence-by-sentence to prevent autoregressive quality degradation.

        Args:
            lookback: If True, merge previous sentence's audio features into
                prompt_cache for each subsequent sentence (improves prosody
                continuity at the cost of slight error propagation).
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        sr = current_model.tts_model.sample_rate

        # Build prompt_cache once (shared across all segments)
        from voxcpm.model.voxcpm2 import VoxCPM2Model
        is_v2 = isinstance(current_model.tts_model, VoxCPM2Model)

        # Handle denoising for prompt/reference audio
        actual_prompt_path = audio_path if (prompt_text_clean and audio_path) else None
        actual_ref_path = audio_path

        if denoise and current_model.denoiser is not None:
            import tempfile
            try:
                if actual_prompt_path:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    current_model.denoiser.enhance(actual_prompt_path, output_path=tmp.name)
                    actual_prompt_path = tmp.name
                if actual_ref_path and actual_ref_path != actual_prompt_path:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    current_model.denoiser.enhance(actual_ref_path, output_path=tmp.name)
                    actual_ref_path = tmp.name
            except Exception as e:
                logger.warning(f"Denoising failed, skipping: {e}")
                actual_prompt_path = audio_path if (prompt_text_clean and audio_path) else None
                actual_ref_path = audio_path

        # Build prompt_cache
        if actual_prompt_path or actual_ref_path:
            if is_v2:
                fixed_prompt_cache = current_model.tts_model.build_prompt_cache(
                    prompt_text=prompt_text_clean if actual_prompt_path else None,
                    prompt_wav_path=actual_prompt_path,
                    reference_wav_path=actual_ref_path if (actual_ref_path and not actual_prompt_path) else (
                        actual_ref_path if (actual_ref_path and actual_prompt_path) else None
                    ),
                )
            else:
                fixed_prompt_cache = current_model.tts_model.build_prompt_cache(
                    prompt_text=prompt_text_clean,
                    prompt_wav_path=actual_prompt_path,
                )
        else:
            fixed_prompt_cache = None

        # Normalize text if needed
        if do_normalize:
            if current_model.text_normalizer is None:
                from voxcpm.utils.text_normalize import TextNormalizer
                current_model.text_normalizer = TextNormalizer()
            # Normalize will be applied per-segment below

        # Prepend control instruction to each sentence
        sentences = self._split_sentences(text)
        mode_str = "lookback-1-clean" if lookback else "clear"
        logger.info(f"Segmented generation ({mode_str}): {len(sentences)} segments")

        from voxcpm.core import next_and_close

        def _gen_one(text_str, cache):
            """Generate one sentence, return (wav_np, audio_feat)."""
            gen = current_model.tts_model._generate_with_prompt_cache(
                target_text=text_str,
                prompt_cache=cache,
                inference_timesteps=inference_timesteps,
                cfg_value=float(cfg_value_input),
                streaming=False,
            )
            wav_t, _, feat = next_and_close(gen)
            return wav_t.squeeze(0).cpu().numpy(), feat

        wav_segments = []
        prev_clean_feat = None  # clean feature from previous sentence (generated with original cache)

        for i, sentence in enumerate(sentences):
            if do_normalize and current_model.text_normalizer is not None:
                sentence = current_model.text_normalizer.normalize(sentence)

            final_sentence = f"({control}){sentence}" if control else sentence
            logger.info(f"  [{i+1}/{len(sentences)}] '{final_sentence[:60]}...'")

            if not lookback or not is_v2 or prev_clean_feat is None:
                # First sentence or no-lookback mode: generate with original cache
                wav_np, clean_feat = _gen_one(final_sentence, fixed_prompt_cache)
                wav_segments.append(wav_np)
                prev_clean_feat = (final_sentence, clean_feat)
            else:
                # Lookback mode (sentence 2+):
                # Pass 1: generate with original cache → clean features (for next sentence)
                logger.info(f"    Pass 1: clean generation (original cache)")
                _, clean_feat = _gen_one(final_sentence, fixed_prompt_cache)

                # Pass 2: generate with original cache + prev clean features → output audio
                lookback_cache = current_model.tts_model.merge_prompt_cache(
                    fixed_prompt_cache,
                    new_text=prev_clean_feat[0],
                    new_audio_feat=prev_clean_feat[1],
                )
                logger.info(f"    Pass 2: lookback generation (+ prev clean feat)")
                wav_np, _ = _gen_one(final_sentence, lookback_cache)
                wav_segments.append(wav_np)
                prev_clean_feat = (final_sentence, clean_feat)

        # Concatenate all segments
        final_wav = np.concatenate(wav_segments)
        logger.info(f"Segmented generation complete: {len(wav_segments)} segments, {len(final_wav)/sr:.1f}s total")
        return (sr, final_wav)


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    # Lazy-loaded audio super-resolution model
    _sr_model = None

    def _get_sr_model():
        nonlocal _sr_model
        if _sr_model is None:
            from audio_sr import AudioSuperResolution
            try:
                _sr_model = AudioSuperResolution(device="cuda")
                logger.info("AP-BWE audio super-resolution model loaded")
            except FileNotFoundError as e:
                logger.warning(f"AP-BWE model not found: {e}")
                return None
        return _sr_model

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        lora_selection: Optional[str] = None,
        audio_sr_enabled: bool = False,
        seg_enabled: bool = False,
        seg_lookback: int = 0,
    ):
        # LoRA hot-swap before generation
        demo.swap_lora(lora_selection)

        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction

        if seg_enabled:
            sr, wav_np = demo.generate_tts_audio_segmented(
                text_input=text,
                control_instruction=actual_control,
                reference_wav_path_input=ref_wav,
                prompt_text=actual_prompt_text,
                cfg_value_input=cfg_value,
                do_normalize=do_normalize,
                denoise=denoise,
                inference_timesteps=int(dit_steps),
                lookback=(int(seg_lookback) >= 1),
            )
        else:
            sr, wav_np = demo.generate_tts_audio(
                text_input=text,
                control_instruction=actual_control,
                reference_wav_path_input=ref_wav,
                prompt_text=actual_prompt_text,
                cfg_value_input=cfg_value,
                do_normalize=do_normalize,
                denoise=denoise,
                inference_timesteps=int(dit_steps),
            )

        # Apply audio super-resolution if enabled
        if audio_sr_enabled:
            sr_model = _get_sr_model()
            if sr_model is not None:
                logger.info("Applying AP-BWE audio super-resolution...")
                wav_np, sr = sr_model(wav_np, orig_sr=sr)
                logger.info(f"Super-resolution complete, output sr={sr}")

        return (sr, wav_np)

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion("🧩 LoRA", open=False):
                    with gr.Row():
                        lora_select = gr.Dropdown(
                            choices=["None"] + scan_lora_checkpoints(),
                            value="None",
                            label="LoRA Checkpoint",
                            scale=4,
                        )
                        refresh_lora_btn = gr.Button("🔄", scale=1, min_width=60)

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )
                    audio_sr_checkbox = gr.Checkbox(
                        value=False,
                        label=I18N("audio_sr_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("audio_sr_info"),
                    )
                    seg_enabled_checkbox = gr.Checkbox(
                        value=False,
                        label=I18N("seg_mode_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("seg_mode_info"),
                    )
                    seg_lookback_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0,
                        step=1,
                        label="回看步数",
                        info="0 = 每句清除 KV Cache，1 = 回看上一句",
                    )
                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"))
                gr.Markdown(I18N("examples_footer"))

        # LoRA refresh
        refresh_lora_btn.click(
            fn=lambda: gr.update(choices=["None"] + scan_lora_checkpoints()),
            outputs=[lora_select],
        )

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        run_btn.click(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                lora_select,
                audio_sr_checkbox,
                seg_enabled_checkbox,
                seg_lookback_slider,
            ],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )

    return interface

def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
):
    demo = VoxCPMDemo(model_id=model_id)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    args = parser.parse_args()
    run_demo(model_id=args.model_id, server_port=args.port)
