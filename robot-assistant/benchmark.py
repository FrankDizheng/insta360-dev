"""Qwen3.5-27B-Opus Intelligence Benchmark"""
import time
from openai import OpenAI

client = OpenAI(base_url="http://8.211.22.21:8094/v1", api_key="none")
MODEL = "qwen3.5-27b-opus"

tests = [
    ("逻辑推理", "一个房间里有3个开关，分别控制隔壁房间的3盏灯。你只能进隔壁房间一次。怎么确定哪个开关控制哪盏灯？"),
    ("代码能力", "写一个Python函数，输入一个列表，返回其中出现次数最多的元素。如果有多个，返回最先出现的那个。"),
    ("数学推理", "一只蜗牛爬一口10米深的井。白天爬3米，晚上滑下2米。请问几天能爬出来？"),
    ("中文理解", "「我差点没考上大学」和「我差点考上大学」，这两句话意思一样吗？解释一下。"),
    ("创意写作", "用一句话描述：一个程序员第一次看到自己写的机器人动起来的心情。"),
    ("常识判断", "把大象放进冰箱需要几步？如果是把长颈鹿放进冰箱呢？"),
    ("角色扮演", "你是一个傲娇的猫娘管家。主人回家了，请回应。"),
    ("多语言", "Translate to English and explain the cultural context: 塞翁失马，焉知非福"),
]

SEP = "-" * 60

print("=" * 60)
print("Qwen3.5-27B-Opus Intelligence Benchmark")
print("=" * 60)

total_tokens = 0
total_time = 0

for i, (category, prompt) in enumerate(tests, 1):
    print(f"\n{SEP}")
    print(f"Test {i}/{len(tests)}: [{category}]")
    print(f"Q: {prompt}")
    print(SEP)

    t0 = time.time()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.6,
    )
    elapsed = time.time() - t0
    content = r.choices[0].message.content
    tokens = r.usage.completion_tokens

    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    print(f"A: {content}")
    print(f"  [{tokens} tokens | {elapsed:.1f}s | {tokens/elapsed:.0f} tok/s]")

    total_tokens += tokens
    total_time += elapsed

print(f"\n{'=' * 60}")
print(f"Total: {total_tokens} tokens in {total_time:.1f}s")
print(f"Average: {total_tokens/total_time:.0f} tok/s")
print("=" * 60)
