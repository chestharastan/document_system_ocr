import torch
from pathlib import Path

ckpt_path = "/home/thareah/Desktop/Khmer-OCR/Model/final2/best_model.pth"   # change to yours
state = torch.load(ckpt_path, map_location="cpu")

# print every key + its shape
for k, v in state.items():
    print(f"{k:60s}  {tuple(v.shape)}")