import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel

# 1.配置
base_model_path=r"F:\Pycharm\大模型医疗微调\model\Qwen\Qwen3-4B-Instruct-2507"
lora_path=r"F:\Pycharm\大模型医疗微调\qwen3_medical_lora"

# 加载tokenizer
tokenizer=AutoTokenizer.from_pretrained(lora_path,trust_remote_code=True)

# 2.量化配置和加载模型
bnb_config=BitsAndBytesConfig(  #量化配置
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)

base_model=AutoModelForCausalLM.from_pretrained(  #加载基础模型
    base_model_path,
    trust_remote_code=True,
    device_map='auto',
    quantization_config=bnb_config,
    dtype=torch.float16
)
model=PeftModel.from_pretrained(base_model,lora_path)   #把lora加载到基础模型里面
model.eval()

history=[]
def chat(user_input):
    history.append({"role":"user","content":user_input})
    prompt=''
    for msg in history:
        prompt+=f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt+=f"<|im_start|>assistant\n"

    inputs=tokenizer(   #输入处理
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=2048).to(model.device)

    outputs=model.generate( #model.generate()是生成文本,model()是前向传播
        **inputs,
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.9,
        do_sample=True
    )
    response=tokenizer.decode(outputs[0],skip_special_tokens=False)
    response=response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>","").strip()
    history.append({"role": "assistant","content":response})

    return response

print("进行医疗对话问答:输入'clear'清空历史,输入'exit'退出")
while True:
    user_input=input("\n我:").strip()
    if user_input.lower() == "clear":
        history=[]
        print("历史已清空!")
        continue
    if user_input.lower()=='exit':
        break
    if user_input:
        print("\n正在思考...")
        print(f"助手:{chat(user_input)}")
