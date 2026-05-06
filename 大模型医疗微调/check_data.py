import json

data_path = r"F:\下载\train_0001_of_0001.json"
print("加载数据中...")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据集大小: {len(data)} 条\n")

# 1. 统计数据集长度
print("="*50)
print("1. 文本长度统计")
print("="*50)
inst_lengths = [len(item['instruction']) for item in data]
out_lengths = [len(item['output']) for item in data]
input_lengths = [len(item['input']) for item in data if item['input']]
print(f"Instruction - 平均:{sum(inst_lengths)/len(inst_lengths):.1f}  范围:[{min(inst_lengths)}-{max(inst_lengths)}]")
print(f"Input       - 平均:{sum(input_lengths)/len(input_lengths) if input_lengths else 0:.1f}    范围:[{min(input_lengths)}-{max(input_lengths)}]")
print(f"Output      - 平均:{sum(out_lengths)/len(out_lengths):.1f}  范围:[{min(out_lengths)}-{max(out_lengths)}]")
# 长度分布
print(f"\nInstruction分布: "
      f"<15字:{sum(1 for l in inst_lengths if l < 15)}条, "
      f"15-30字:{sum(1 for l in inst_lengths if 15 <= l <= 30)}条, "
      f">30字:{sum(1 for l in inst_lengths if l > 30)}条"
      )

# 2. 数据质量检查
print("\n" + "="*50)
print("2. 数据质量检查")
print("="*50)
empty_input = sum(1 for item in data if not item['input'])
short_output = sum(1 for item in data if len(item['output']) < 50)
long_output = sum(1 for item in data if len(item['output']) > 2000)
print(f"空Input: {empty_input}条 ({empty_input/len(data)*100:.2f}%)")
print(f"过短Output(<50字): {short_output}条 ({short_output/len(data)*100:.2f}%)")
print(f"过长Output(>2000字): {long_output}条 ({long_output/len(data)*100:.2f}%)")

# 3. 重复检查
print("\n" + "="*50)
print("3. 重复检查")
print("="*50)
# 检查重复：检查 (instruction, input, output) 是否已经存在
seen = set()
duplicate_count = 0  # 记录重复的数据
for item in data:
    key = (item['instruction'], item['input'], item['output'])
    if key in seen:
        duplicate_count += 1
    else:
        seen.add(key)
# 打印重复统计
print(f"总数据: {len(data)}条")
print(f"重复数据: {duplicate_count}条")
print(f"唯一数据: {len(seen)}条")

# 4. 展示样例
print("\n" + "="*50)
print("4. 数据样例")
print("="*50)
for i in range(3):
    print(f"\n样例{i+1}:")
    print(f"  instruction: {data[i]['instruction']}")
    print(f"  input: {data[i]['input'][:50]}...")
    print(f"  output: {data[i]['output'][:100]}...")