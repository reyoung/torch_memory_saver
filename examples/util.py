import subprocess


def print_gpu_memory_gb(message, gpu_id=0):
    """Print GPU memory usage with optional message"""
    cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id={gpu_id}"
    result = subprocess.check_output(cmd, shell=True, text=True).strip()
    print(f"GPU {gpu_id} memory: {int(result) / 1024:.2f} GB ({message})")
