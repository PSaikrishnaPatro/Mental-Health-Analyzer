import gdown
import os

output_path = "model/model.safetensors"

# Skip download if file already exists
if os.path.exists(output_path) and os.path.getsize(output_path) > 1_000_000:
    print(f"Model already exists at '{output_path}'. Skipping download.")
else:
    file_id = "1rcN8X8PLUOlm5fGvRzj3lwfnZytdqrsi"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Downloading model weights from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    print(f"Model downloaded successfully to '{output_path}'")
