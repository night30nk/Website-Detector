# try:
#     import torch
#     print(f"PyTorch is available. Version: {torch.__version__}")
#
#     # Check if CUDA (GPU support) is available
#     if torch.cuda.is_available():
#         print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         print("CUDA is not available. Using CPU.")
#
# except ImportError:
#     print("PyTorch is not installed in this environment.")

from features import extract_features
features = extract_features("http://chatgpt.com")
print(features)