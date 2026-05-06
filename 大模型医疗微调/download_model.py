from modelscope import snapshot_download

model_dir =snapshot_download(
    'Qwen/Qwen3-4B-Instruct-2507',
    cache_dir='./model',
    revision='master'
)
print(f"模型已下载到{model_dir}")
