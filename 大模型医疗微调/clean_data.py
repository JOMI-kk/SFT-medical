import json
import random

random.seed(15)
# ==============================1.清洗数据==============================
# 加载数据
print("加载数据...")
with open(r"F:\下载\train_0001_of_0001.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始数据: {len(data)}条")

# 清洗 + 过滤input="无"的数据 + 去重
print("处理中...")
final_data = []
seen = set()
filtered_count = 0  # 记录被过滤掉的数据（质量差 + input="无"）
duplicate_count = 0  # 记录被去重移除的数据

for item in data:
    # 清洗条件
    if len(item['output']) < 50:
        filtered_count += 1
        continue
    if len(item['output']) > 2000:
        filtered_count += 1
        continue
    if len(item['instruction']) < 5:
        filtered_count += 1
        continue

    # 过滤input为"无"的数据
    if item['input'] == "无" or not item['input'].strip():
        filtered_count += 1
        continue

    # 去重：检查 (instruction, input, output) 是否已经存在
    key = (item['instruction'], item['input'], item['output'])
    if key in seen:
        duplicate_count += 1
        continue
    seen.add(key)

    # 收集数据
    final_data.append(item)

    # 达到20万条就停止
    if len(final_data) >= 200000:
        break

print(f"清洗后: {len(final_data)}条")
print(f"质量过滤(输出过短/过长/指令过短/input='无'): {filtered_count} 条")
print(f"完全去重移除: {duplicate_count} 条")
print(f"总处理数据: {len(final_data) + filtered_count + duplicate_count} 条")

# 保存
with open('train_cleaned_200k.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print("保存成功: train_cleaned_200k.json")

# 验证一下
print("\n验证结果:")
print(f"最后一条数据的input: {final_data[-1]['input']}")

# ==============================2.将数据转化为qwen对话格式===============================

print("==============================将数据转化为qwen对话格式===============================")
print("加载数据...")
with open('train_cleaned_200k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"总数据量: {len(data)}条")

# 转换为Qwen对话格式
print("转换为Qwen格式...")
formatted_data = []

for item in data:
    # 构建Qwen标准对话格式
    dialogue = {
        "conversations": [
            {
                "role": "system",
                "content": "你是一位专业的医疗助手，请根据患者描述提供准确、专业的医疗建议。如果情况严重，建议及时就医。"
            },
            {
                "role": "user",
                "content": f"{item['instruction']}\n\n患者情况：{item['input']}" if item['input'] else item['instruction']
            },
            {
                "role": "assistant",
                "content": item['output']
            }
        ]
    }
    formatted_data.append(dialogue)

# 打乱数据
random.shuffle(formatted_data)

# 划分数据集 (80%训练, 10%验证, 10%测试)
total = len(formatted_data)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_data = formatted_data[:train_end]
val_data = formatted_data[train_end:val_end]
test_data = formatted_data[val_end:]

print(f"\n数据集划分完成:")
print(f"训练集: {len(train_data)}条")
print(f"验证集: {len(val_data)}条")
print(f"测试集: {len(test_data)}条")

# 保存为Qwen格式
print("\n保存数据...")
with open('qwen_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('qwen_val.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open('qwen_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(" 保存完成！")
print("   - qwen_train.json (训练集)")
print("   - qwen_val.json (验证集)")
print("   - qwen_test.json (测试集)")

# 打印样例验证
print("\n样例数据预览:")
sample = train_data[0]
for s in sample['conversations']:
    print(f"\n{s['role']}:")
    print(f"  {s['content']}")