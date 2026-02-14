from huggingface_hub import snapshot_download
import os

# Create a local folder for the model
local_dir = "tinyllama_local"
os.makedirs(local_dir, exist_ok=True)

print("Downloading TinyLlama weights from Hugging Face...")
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir=local_dir,
    local_dir_use_symlinks=False, # Saves actual files, not links
    ignore_patterns=["*.msgpack", "*.h5", "model.safetensors"] # Download the .bin or .pt files
)
print(f"Model saved to {local_dir}")