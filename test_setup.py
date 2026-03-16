import torch
import os
from pathlib import Path

print("PyTorch 테스트")
print(f"PyTorch 버전: {torch.__version__}")
print(f"GPU 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
else:
    print("CPU 모드로 실행됩니다")

# 데이터 확인
data_path = Path("data/raw/train")
print(f"\n데이터 경로: {data_path}")
print(f"경로 존재: {data_path.exists()}")

if data_path.exists():
    cats = list((data_path / "cats").glob("*.jpg"))
    dogs = list((data_path / "dogs").glob("*.jpg"))
    print(f"Cats 이미지: {len(cats)}")
    print(f"Dogs 이미지: {len(dogs)}")
    
    if cats:
        print(f"샘플 cat 이미지: {cats[0].name}")
    if dogs:
        print(f"샘플 dog 이미지: {dogs[0].name}")
