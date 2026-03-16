# Dogs vs Cats Image Classification

개/고양이 이미지를 구분하는 PyTorch 기반 이미지 분류 서비스다. 추론 API는 FastAPI, 웹 UI는 Streamlit으로 구성되어 있고, Grad-CAM 시각화를 지원한다.

# Overview

이미지를 업로드하면 모델이 dog/cat으로 예측하고, 원하는 경우 Grad-CAM으로 어떤 영역을 보고 판단했는지 볼 수 있다.

```
User
  ↓
Streamlit UI
  ↓
FastAPI
  ↓
PyTorch Model (EfficientNet-B0)
  ↓
Prediction + Grad-CAM
```

# Features

- Dog / Cat 이진 분류
- FastAPI 추론 API (`/predict`, `/explainability` 등)
- Grad-CAM 시각화 (dog/cat 예측 시)
- 신뢰도/거리 기반 unknown 처리 (개·고양이 외 이미지 거부)
- 이미지 입력 검증
- Streamlit 웹 UI

# Project Structure

- **app/** — FastAPI 서버 (inference, Grad-CAM, 모델 로딩, 라우트)
- **config/** — 설정 파일 (config.yaml, training_config.yaml)
- **models/** — 학습된 가중치·센트로이드 (저장소에는 없을 수 있음)
- **scripts/** — 학습·데이터 정리 스크립트 (train_simple.py, organize_data.py)
- **train/** — 학습용 유틸 (dataset, evaluate, compute_centroids 등)
- **ui/** — Streamlit 앱 (streamlit_app.py)

# Model

- **구조**: EfficientNet-B0 (timm), 2-class 분류
- **데이터**: Kaggle Dogs vs Cats (train: cats/, dogs/ 폴더 구조)
- **전처리**: 224x224 리사이즈, ImageNet mean/std 정규화
- **출력**: label (dog / cat / unknown), confidence (0~1)

# API

**POST /predict**

이미지를 base64로 인코딩해 JSON body로 보내면, 예측 결과와 신뢰도를 JSON으로 받는다.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-encoded-image>"}'
```

응답 예:

```json
{
  "label": "dog",
  "confidence": 0.95,
  "status": "success",
  "processing_time_ms": 45.2
}
```

Grad-CAM이 필요하면 **POST /explainability** 를 호출하면 된다. (dog/cat 예측일 때만 사용 가능)

# Run Locally

의존성 설치:

```bash
pip install -r requirements.txt
```

FastAPI 서버 실행 (프로젝트 루트에서):

```bash
uvicorn app.main:app --reload
```

다른 터미널에서 Streamlit UI 실행:

```bash
streamlit run ui/streamlit_app.py
```

API는 http://localhost:8000, UI는 http://localhost:8501 에서 접속한다. 서버 기동 전에 `models/checkpoints/best_model.pth` 가 있어야 한다.

# Tech Stack

- Python
- PyTorch, torchvision, timm
- FastAPI, Uvicorn
- Streamlit
- OpenCV, Pillow
- PyYAML, MLflow (학습/설정용)

# Notes

- 학습된 모델 가중치(`models/checkpoints/best_model.pth`)와 데이터셋은 저장소에 포함되지 않을 수 있다. 직접 학습하거나 체크포인트를 해당 경로에 두면 된다.
- OOD용 센트로이드(`models/centroids.npz`)는 선택 사항이다. 없으면 신뢰도 임계값만으로 unknown을 판단한다.
