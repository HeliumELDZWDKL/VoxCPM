import sys
print("Python:", sys.version)
try:
    import torch
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
except ImportError:
    print("PyTorch: NOT INSTALLED")

deps = ["voxcpm", "argbind", "tensorboardX", "safetensors", "soundfile", "funasr", "gradio", "yaml"]
for d in deps:
    try:
        m = __import__(d)
        v = getattr(m, "__version__", "ok")
        print(f"  {d}: {v}")
    except ImportError:
        print(f"  {d}: MISSING")
