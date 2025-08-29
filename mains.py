import platform
import torch
import os

print("===== SYSTEM INFO =====")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python Version: {platform.python_version()}")

print("\n===== TORCH INFO =====")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Running on CPU.")

print("\n===== MEMORY INFO (Linux Only) =====")
if os.name == "posix":
    os.system("free -h")
