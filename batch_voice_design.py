"""
批量音色设计生成：为木清灵寻找合适音色。
使用音色设计模式（无参考音频），固定 prompt 和 target text，
通过不同随机种子生成 100 个候选音频供挑选。
"""

import os
import sys
import time
import soundfile as sf
import torch
import numpy as np

# ── 配置区 ──────────────────────────────────────────────
CONTROL_INSTRUCTION = "二次元女角色，温柔，气声，亲和。"
TARGET_TEXT = "你好像有不少问题想问我呢！别心急，我们还有很多很多时间~"

CFG_VALUE = 1.4
LOCDIT_STEPS = 50
DENOISE = False
NORMALIZE = False

TOTAL_COUNT = 100
OUTPUT_DIR = r"E:\SVN_灵气\语音合成\关键留档\木清灵_甘雨琴\备选"


# ── 主逻辑 ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== VoxCPM2 批量音色设计 ===", file=sys.stderr)
    print(f"Control: {CONTROL_INSTRUCTION}", file=sys.stderr)
    print(f"Target:  {TARGET_TEXT}", file=sys.stderr)
    print(f"CFG: {CFG_VALUE}, LocDiT Steps: {LOCDIT_STEPS}", file=sys.stderr)
    print(f"共 {TOTAL_COUNT} 个候选音频", file=sys.stderr)
    print(f"输出目录: {OUTPUT_DIR}", file=sys.stderr)
    print(f"", file=sys.stderr)

    # 加载模型
    print("正在加载 VoxCPM2 模型...", file=sys.stderr)
    from src.voxcpm.core import VoxCPM

    model = VoxCPM(
        voxcpm_model_path="models/VoxCPM2",
        enable_denoiser=False,
        optimize=True,
    )
    print("模型加载完成！\n", file=sys.stderr)

    final_text = f"({CONTROL_INSTRUCTION}){TARGET_TEXT}"

    success_count = 0
    failed = []

    for i in range(1, TOTAL_COUNT + 1):
        filename = f"voice_design_{i:03d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"[{i:03d}/{TOTAL_COUNT}] 生成中...", end="", file=sys.stderr)

        try:
            # 每次生成前设置不同的随机种子，确保音色多样性
            seed = i * 1000 + 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            t0 = time.time()
            wav = model.generate(
                text=final_text,
                cfg_value=CFG_VALUE,
                inference_timesteps=LOCDIT_STEPS,
                normalize=NORMALIZE,
                denoise=DENOISE,
            )
            elapsed = time.time() - t0
            duration = len(wav) / model.tts_model.sample_rate

            sf.write(filepath, wav, model.tts_model.sample_rate)
            success_count += 1

            print(f" ✓ {duration:.1f}s 音频, 耗时 {elapsed:.1f}s, seed={seed} → {filename}", file=sys.stderr)

        except Exception as e:
            print(f" ✗ 失败: {e}", file=sys.stderr)
            failed.append((i, str(e)))

    print(f"\n=== 完成 ===", file=sys.stderr)
    print(f"成功: {success_count}/{TOTAL_COUNT}", file=sys.stderr)
    print(f"失败: {len(failed)}", file=sys.stderr)
    print(f"音频目录: {OUTPUT_DIR}", file=sys.stderr)

    if failed:
        print(f"\n失败列表:", file=sys.stderr)
        for idx, err in failed:
            print(f"  [{idx:03d}] {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
