# 모델 비교 보고서 (학습·추론 로그 기반)

Dog vs Cat 분류 프로젝트에서 사용 가능한 백본 모델(ResNet18, EfficientNet-B0, ConvNeXt-Tiny)의 **실제 로그 기반** 성능 요약입니다.

---

## 1. 데이터 출처

| 출처 | 설명 |
|------|------|
| `logs/inference.log` | API 서버 기동 시 체크포인트 로드 메시지에 기록된 `val_acc` |
| `models/checkpoints/best_model.pth` | 훈련 시 저장된 체크포인트(메타데이터에 `val_acc` 포함) |
| `config/config.yaml` | 현재 프로덕션 백본: `efficientnet_b0` |

현재 저장소에는 **EfficientNet-B0** 한 종류만 학습되어 체크포인트와 로그에 기록되어 있습니다. ResNet18, ConvNeXt-Tiny는 동일 파이프라인으로 학습 후 아래 표를 보완할 수 있습니다.

---

## 2. 모델별 성능 요약

### 2.1 검증 정확도 (Validation Accuracy)

| 백본 모델 | Val Accuracy | 출처 | 비고 |
|-----------|--------------|------|------|
| **EfficientNet-B0** | **99.08%** | `logs/inference.log` (체크포인트 로드 시 로그) | 현재 프로덕션 사용 모델 |
| ResNet18 | — | (미학습) | 동일 설정 학습 후 기록 |
| ConvNeXt-Tiny | — | (미학습) | 동일 설정 학습 후 기록 |

- EfficientNet-B0 수치: `Model loaded from ... best_model.pth, val_acc=0.9908` 에서 추출 (실제 로그 기준).

### 2.2 로그 발췌 (EfficientNet-B0)

```
app.model_loader - INFO - Model loaded from models\checkpoints\best_model.pth, val_acc=0.9908
```

---

## 3. 실험 재현 방법 (다중 모델 비교용)

다른 백본으로 학습해 위 표를 채우려면 아래 순서로 진행하면 됩니다.

1. **백본 변경**  
   `config/config.yaml` 의 `model.backbone` 을 다음 중 하나로 설정:
   - `resnet18`
   - `efficientnet_b0`
   - `convnext_tiny`

2. **훈련 실행**  
   `train.py` 는 상대 경로(`../config`, `../data`)를 사용하므로 **반드시 `train` 디렉터리에서** 실행합니다.
   ```bash
   cd train
   python train.py
   ```
   (프로젝트 루트에서 `cd train` 후 실행)

3. **결과 확인**
   - 체크포인트: `models/checkpoints/best_model.pth` (덮어쓰기 되므로 백본별로 백업 권장)
   - MLflow: `./mlruns` 에 `best_val_acc` 등 메트릭 기록
   - 서버 재기동 시 `logs/inference.log` 에 `val_acc=...` 로 출력

4. **표 보완**  
   각 백본 학습 후 `logs/inference.log` 또는 MLflow에서 `best_val_acc` / `val_acc` 를 확인해 위 표에 추가하면 됩니다.

---

## 4. 공통 학습 설정 (동일 조건 비교용)

아래 설정으로 학습 시 모델 간 공정 비교가 가능합니다.

| 항목 | 값 | 설정 파일 |
|------|-----|-----------|
| epochs | 30 | `config/training_config.yaml` |
| batch_size | 64 | `config/training_config.yaml` |
| learning_rate | 0.0003 | `config/training_config.yaml` |
| optimizer | AdamW | `config/training_config.yaml` |
| scheduler | cosine | `config/training_config.yaml` |
| pretrained | true | `config/config.yaml` |
| num_classes | 2 | `config/config.yaml` |
| dropout_rate | 0.5 | `config/config.yaml` |

---

## 5. 요약

- **실제 로그 기반 수치**: EfficientNet-B0 **Val Accuracy 99.08%** (체크포인트 로드 로그 기준).
- ResNet18, ConvNeXt-Tiny는 동일 파이프라인으로 학습 후 `logs/inference.log` 및 MLflow 결과를 이 문서에 반영하면 됩니다.
- 실험 재현은 `config/config.yaml` 의 `backbone` 변경 후 `train/train.py` 실행으로 가능합니다.
