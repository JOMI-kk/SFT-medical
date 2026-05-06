import os
import glob

# 定义模型路径
model_path = r"F:\Pycharm\大模型医疗微调\model\Qwen\Qwen3-4B-Instruct-2507"

# 定义要查找的常见配置文件模式
config_patterns = [
    "config.json",  # 模型配置文件
    "tokenizer_config.json",  # 分词器配置文件
    "tokenizer.json",  # 分词器文件 (Fast Tokenizer)
    "vocab.json",  # 词汇表文件 (例如 GPT-2, Roberta)
    "merges.txt",  # BPE 合并规则文件 (例如 GPT-2, Roberta)
    "special_tokens_map.json",  # 特殊 token 映射文件
]

print(f"正在检查模型路径: {model_path}\n")

# 检查每个模式
found_files = {}
missing_files = []

for pattern in config_patterns:
    # 使用 glob 查找文件
    full_pattern = os.path.join(model_path, pattern)
    matches = glob.glob(full_pattern)

    if matches:
        found_files[pattern] = matches
        print(f"✅ 找到: {pattern} -> {matches[0]}")  # 只打印第一个匹配项
    else:
        missing_files.append(pattern)
        print(f"❌ 未找到: {pattern}")

print("\n" + "=" * 50)
print("检查总结:")
print("=" * 50)

if found_files:
    print(f"\n找到的配置文件 ({len(found_files)} 个):")
    for filename, paths in found_files.items():
        for path in paths:
            print(f"  - {path}")
else:
    print("\n未找到任何配置文件。")

if missing_files:
    print(f"\n缺失的配置文件 ({len(missing_files)} 个):")
    for filename in missing_files:
        print(f"  - {filename}")

    if 'config.json' in missing_files:
        print("\n[重要] 缺少 'config.json' 文件，这很可能是 PEFT 发出警告的原因。")
else:
    print("\n所有预期的配置文件都已找到。")

print("\n检查完成。")