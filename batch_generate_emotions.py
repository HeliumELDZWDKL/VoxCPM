"""
批量生成多情绪语音数据，用于 VoxCPM2 LoRA 训练。
使用音色设计模式，固定 Control Instruction，生成 30 段不同情绪方向的语音。
"""

import os
import sys
import json
import time
import soundfile as sf

# ── 配置区 ──────────────────────────────────────────────
CONTROL_INSTRUCTION = (
    "清澈温婉的少御音、干净的少女声、中高音色。"
    "情温柔、礼貌、沉稳、略显羞涩、治愈系。"
    "气息感重、语速平缓、吐字清晰但轻柔"
)

CFG_VALUE = 1.6
LOCDIT_STEPS = 20
DENOISE = False
NORMALIZE = False

OUTPUT_DIR = "lora_train_data"
JSONL_PATH = os.path.join(OUTPUT_DIR, "train_manifest.jsonl")

# ── 30 段不同情绪方向的文本（每段约 50 字）──────────────
EMOTION_TEXTS = [
    # 1-5: 温柔 / 治愈
    "没关系的，慢慢来就好。不管发生了什么，我都会在你身边，陪着你一起走过这段路。你从来都不是一个人。",
    "今天辛苦了吧？回来的路上有没有看到那棵开了花的树？春天来了呢，一切都会慢慢好起来的，你放心。",
    "你知道吗，每次听到你的声音，我就觉得心里很安定。就像冬天里捧着一杯热可可，暖暖的，很幸福。",
    "如果你累了，就休息一下吧。不用总是逞强，偶尔示弱也没关系的。我在这里，不会走的，等你就好。",
    "晚安呀，今天也请好好休息。做一个甜甜的梦，梦里有星星和月亮，还有一个一直守护你的人。",

    # 6-10: 开心 / 甜美
    "太好了！我等这一天等了好久呢！你终于答应了，我真的好开心呀，开心到想转圈圈！谢谢你！",
    "哇，你今天做的蛋糕好好吃呀！软软的，甜甜的，是我最喜欢的草莓味！下次还可以一起做吗？",
    "生日快乐呀！我偷偷给你准备了一个小礼物，你猜猜是什么？嘿嘿，等你打开就知道了！",
    "今天天气好好呀，阳光暖暖的，微风轻轻的。我们去公园散步好不好？可以一起喂小鸭子呢！",
    "你看你看，那边有一只好可爱的小猫咪！毛茸茸的，眼睛圆圆的。好想摸摸它呀，它在冲我喵了！",

    # 11-15: 害羞 / 忐忑
    "那个，我想跟你说一件事，但是有点不好意思。就是，嗯，你能不能再靠近我一点点？只要一点点就好。",
    "谢谢你的夸奖，我其实没有你说的那么好啦。被你这样看着，我的脸好烫啊，你不要再盯着我了嘛。",
    "我第一次做这个，不知道味道怎么样。你先尝一口好不好？如果不好吃的话，也请你不要嫌弃我。",
    "你有没有发现，每次和你说话的时候，我的心跳都会变得好快。大概是因为你太温柔了吧，我好紧张。",
    "那封信你看了吗？里面写的都是真心话。如果你觉得奇怪的话就当我没说好了，不要讨厌我就好。",

    # 16-20: 认真 / 沉稳
    "这件事情需要我们认真对待，不能马虎。我已经把所有资料整理好了，你看一下，有问题随时告诉我。",
    "我理解你的顾虑，但我们必须做出选择。每条路都有风险，关键是我们一起面对，就不会有什么问题。",
    "请你听我说完，这很重要。我反复确认过了，这个方案是目前最稳妥的，细节我都仔细推敲过了。",
    "学习这件事没有捷径，需要一步一个脚印地走。今天的积累，明天一定会看到回报的，我相信你。",
    "作为你的搭档，我会尽全力支持你。有什么困难尽管告诉我，我们一起想办法，总能找到解决的。",

    # 21-25: 忧伤 / 感伤
    "窗外又下雨了，空气里弥漫着泥土的气息。不知道为什么，每到这种天气，我就会特别想念从前。",
    "有些话藏在心里太久了，久到我自己都快忘了。可是偶尔午夜梦回，那些记忆还是会悄悄涌上来。",
    "再见这个词好残忍啊，明明是祝福，听起来却像告别。如果可以的话，我多希望时间能停在那一刻。",
    "你走了以后，这条路变得好安静。以前我们一起走的时候，总觉得路太短了，现在却觉得走不到尽头。",
    "我把那张照片收好了，放在日记本的最后一页。偶尔翻到的时候，我会笑一笑，然后轻轻合上。",

    # 26-30: 鼓励 / 打气
    "别怕，你已经做得很好了。比起昨天的自己，你进步了好多呢。每一步都算数的，我一直在看着你。",
    "就算失败了也没关系呀，失败说明你在尝试。很多人连迈出第一步的勇气都没有呢，你已经很厉害了。",
    "你一定可以的，我从来没有怀疑过你。就像种子总会发芽一样，你的努力一定会开出最美的花来。",
    "深呼吸，放轻松。不管结果怎样，我都为你骄傲。你的勇气和坚持，本身就是最了不起的事情。",
    "世界上没有白走的路，也没有白吃的苦。总有一天你会感谢现在拼命的自己，到时候记得笑着回头。",
]

EMOTION_LABELS = [
    "温柔治愈", "温柔治愈", "温柔治愈", "温柔治愈", "温柔治愈",
    "开心甜美", "开心甜美", "开心甜美", "开心甜美", "开心甜美",
    "害羞忐忑", "害羞忐忑", "害羞忐忑", "害羞忐忑", "害羞忐忑",
    "认真沉稳", "认真沉稳", "认真沉稳", "认真沉稳", "认真沉稳",
    "忧伤感伤", "忧伤感伤", "忧伤感伤", "忧伤感伤", "忧伤感伤",
    "鼓励打气", "鼓励打气", "鼓励打气", "鼓励打气", "鼓励打气",
]

# ── 主逻辑 ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== VoxCPM2 批量情绪语音生成 ===", file=sys.stderr)
    print(f"Control: {CONTROL_INSTRUCTION[:40]}...", file=sys.stderr)
    print(f"CFG: {CFG_VALUE}, LocDiT Steps: {LOCDIT_STEPS}", file=sys.stderr)
    print(f"共 {len(EMOTION_TEXTS)} 段文本", file=sys.stderr)
    print(f"输出目录: {OUTPUT_DIR}", file=sys.stderr)
    print(f"", file=sys.stderr)

    # 加载模型
    print("正在加载 VoxCPM2 模型...", file=sys.stderr)
    from src.voxcpm.core import VoxCPM

    model = VoxCPM(
        voxcpm_model_path="models/VoxCPM2",
        enable_denoiser=False,  # 禁用降噪，保护48kHz高频细节
        optimize=True,
    )
    print("模型加载完成！\n", file=sys.stderr)

    results = []
    failed = []

    for i, (text, emotion) in enumerate(zip(EMOTION_TEXTS, EMOTION_LABELS), 1):
        filename = f"{i:02d}_{emotion}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"[{i:02d}/30] {emotion} — {text[:30]}...", file=sys.stderr)

        try:
            t0 = time.time()
            final_text = f"({CONTROL_INSTRUCTION}){text}"
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

            results.append({
                "audio": os.path.abspath(filepath),
                "text": text,
                "emotion": emotion,
                "duration": round(duration, 2),
            })

            print(f"       ✓ {duration:.1f}s 音频, 耗时 {elapsed:.1f}s → {filename}", file=sys.stderr)

        except Exception as e:
            print(f"       ✗ 失败: {e}", file=sys.stderr)
            failed.append((i, text, str(e)))

    # 写入 JSONL 训练清单
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in results:
            entry = {"audio": r["audio"], "text": r["text"]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n=== 完成 ===", file=sys.stderr)
    print(f"成功: {len(results)}/30", file=sys.stderr)
    print(f"失败: {len(failed)}", file=sys.stderr)
    print(f"训练清单: {JSONL_PATH}", file=sys.stderr)
    print(f"音频目录: {OUTPUT_DIR}/", file=sys.stderr)

    if failed:
        print(f"\n失败列表:", file=sys.stderr)
        for idx, txt, err in failed:
            print(f"  [{idx:02d}] {txt[:20]}... → {err}", file=sys.stderr)

    # 统计
    total_duration = sum(r["duration"] for r in results)
    print(f"\n总音频时长: {total_duration:.1f}s ({total_duration/60:.1f}分钟)", file=sys.stderr)


if __name__ == "__main__":
    main()
