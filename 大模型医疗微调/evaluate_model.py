# eval_model.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from bert_score import BERTScorer
from tqdm import tqdm

# 1. 加载模型
base_model_path = r"F:\Pycharm\大模型医疗微调\model\Qwen\Qwen3-4B-Instruct-2507"
lora_path = r"./qwen3_medical_lora"

tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # 批量推理必须设置为 left！

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=bnb_config,
                                                  trust_remote_code=True, device_map='auto')
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# 2. 加载测试集
with open("qwen_test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)


# 3. 批量推理函数（添加进度条）
def batch_chat(user_inputs, batch_size=4):
    responses = []
    total_batches = (len(user_inputs) + batch_size - 1) // batch_size

    # 添加进度条
    for i in tqdm(range(0, len(user_inputs), batch_size),
                  desc="批量推理进度",
                  total=total_batches,
                  unit="batch"):
        batch_inputs = user_inputs[i:i + batch_size]

        prompts = []
        for user_input in batch_inputs:
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            response = response.split("<|im_start|>assistant\n")[-1].strip()
            responses.append(response)

    return responses


# 4. 准备数据
test_samples = test_data
user_inputs = [item["conversations"][1]["content"] for item in test_samples]
references = [item["conversations"][2]["content"] for item in test_samples]

# 5. 批量预测
print(f"开始批量预测，共 {len(user_inputs)} 条，批次大小=8")
predictions = batch_chat(user_inputs, batch_size=8)

# 6. 计算 BERTScore
print("计算 BERTScore 中...")
scorer = BERTScorer(lang="zh", model_type="bert-base-chinese")
P, R, F1 = scorer.score(predictions, references)

# 7. 输出结果
print("\n" + "=" * 50)
print("语义相似度评估结果")
print("=" * 50)
print(f"总样本数: {len(predictions)}")
print(f"平均 Precision: {P.mean():.4f}")
print(f"平均 Recall: {R.mean():.4f}")
print(f"平均 F1分数: {F1.mean():.4f}")
print("=" * 50)