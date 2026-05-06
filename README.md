# SFT-medical
# 医疗大模型微调项目 (Medical LLM Fine-tuning)

基于 Qwen3-4B 的医疗问答模型微调，使用 LoRA + 4bit 量化技术。

## 项目简介

本项目使用 LoRA 方法对 Qwen3-4B-Instruct 模型进行微调，使其具备医疗健康领域的问答能力。适用于医疗咨询、健康建议等场景。

## 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft bitsandbytes
pip install bert-score tqdm
```

## 数据格式

训练数据采用 Qwen 对话模板格式：

```json
{
  "conversations": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
  ]
}
```

## 训练

```bash
python train.py
```

训练配置：
- 基础模型：Qwen3-4B-Instruct
- 微调方法：LoRA + 4bit 量化
- 学习率：2e-4
- 批次大小：2（梯度累积 8 步）
- 训练轮数：1

## 评估

```bash
python evaluate_model.py
```

评估指标：BERTScore（语义相似度）

测试结果（5000 条）：
| 指标 | 分数 |
|------|------|
| Precision | 0.6329 |
| Recall | 0.6451 |
| F1 | 0.6385 |

## 交互测试

```bash
python test_interactive.py
```

支持多轮对话记忆，输入 `clear` 清空历史，`exit` 退出。

## 常见问题

### 1. Hugging Face 连接超时
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 2. 批量推理左填充警告
```python
tokenizer.padding_side = 'left'
```

### 3. PyTorch GPU 版本
使用 CUDA 11.8 版本：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 项目结构

```
├── train.py                 # 训练脚本
├── evaluate_model.py        # 评估脚本
├── chat.py      # 交互测试脚本
├── lora_config.py           # LoRA 配置
├── qwen_train.json          # 训练集
├── qwen_val.json            # 验证集
├── qwen_test.json           # 测试集
└── qwen3_medical_lora/      # 训练好的 LoRA 权重
```

## 技术栈

- PyTorch + Transformers
- PEFT (LoRA)
- BitsAndBytes (4bit 量化)
- BERTScore (语义评估)
