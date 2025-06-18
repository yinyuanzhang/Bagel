from huggingface_hub import snapshot_download
import os # 导入 os 模块以确保目录结构正确

save_dir = "/users/zyy/data/work_dir/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = os.path.join(save_dir, "cache") # 使用 os.path.join 来构建路径

# 确保 save_dir 和 cache_dir 存在
os.makedirs(save_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)


print(f"Downloading model from {repo_id} to {save_dir}...")

snapshot_download(cache_dir=cache_dir,
                  local_dir=save_dir,
                  repo_id=repo_id,
                  local_dir_use_symlinks=False,
                  resume_download=True,
                  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

print("Download complete!")