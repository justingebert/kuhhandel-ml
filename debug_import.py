
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("Attempting to import torch...")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print("Torch import successful.")
except ImportError as e:
    print(f"Torch import failed: {e}")
except OSError as e:
    print(f"Torch DLL load failed: {e}")

print("Attempting to import sb3_contrib...")
try:
    from sb3_contrib import MaskablePPO
    print("sb3_contrib import successful.")
except ImportError as e:
    print(f"sb3_contrib import failed: {e}")
except OSError as e:
    print(f"sb3_contrib DLL load failed: {e}")
