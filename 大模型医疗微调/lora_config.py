from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training

# ==================== LoRA 配置参数 ====================
def apply_lora(model):
    """对模型应用LoRA配置"""
    print("="*50)
    print("应用LoRA配置")
    print("=" * 50)
    model=prepare_model_for_kbit_training(model)

    # 创建LoRA配置
    lora_config=LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        task_type="CAUSAL_LM",
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    # 应用LoRA
    model=get_peft_model(model,lora_config)
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params=sum(p.numel() for p in model.parameters() )
    print("LoRA配置完成！")
    print(f"可训练参数：{trainable_params} ({100*trainable_params/total_params:.2f}%)")
    print(f"总参数：{total_params}")
    print(f"LoRA秩{lora_config.r}")
    print(f"目标模块：{len(lora_config.target_modules)}个")

    return model