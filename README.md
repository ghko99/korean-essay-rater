# Korean Essay Rater — AI 한국어 에세이 자동 채점 시스템

FastAPI + vLLM 기반의 한국어 에세이 자동 채점 웹 애플리케이션입니다.
Llama 기반 모델에 LoRA 어댑터를 결합하여 에세이를 8개 루브릭 항목으로 채점하고, 상세 피드백을 실시간 스트리밍으로 제공합니다.
<img width="1078" height="896" alt="image" src="https://github.com/user-attachments/assets/4cff0e1c-a58d-4276-89ba-08e59886f890" />
<img width="904" height="918" alt="image" src="https://github.com/user-attachments/assets/5a111cbe-4f6d-4513-8d16-e60d62461973" />

---

## 주요 기능

- **8개 루브릭 채점** (각 1–9점): 과제 수행의 충실성 / 설명의 명료성 / 구체성 / 적절성 / 문장 연결성 / 글의 통일성 / 어휘 적절성 / 어법 적절성
- **앙상블 채점**: 동일 프롬프트를 50회 샘플링하여 평균 점수 및 분포 산출
- **확률 분포 시각화**: 그리디 패스의 logprobs를 활용해 각 항목별 점수 확률 분포를 차트로 표시
- **형태소 분석**: kiwipiepy 기반 실시간 형태소 분석 및 언어적 특징 추출
- **SSE 스트리밍**: 채점 과정(특징 분석 → 토큰 생성 → 최종 점수)을 실시간 스트림으로 전달

## 시스템 구성

```
사용자 브라우저
    │  SSE / REST API
    ▼
FastAPI (app.py)
    ├── kiwipiepy          형태소 분석 · 언어 특징 추출
    └── vLLM AsyncEngine   Llama + LoRA 어댑터 추론
```

## 요구 사항

- Python 3.10+
- CUDA 지원 GPU (vLLM 실행에 필요)
- base model : meta-llama/Llama-3.1-8B-Instruct (별도 다운로드 필요)

## 설치

### 1. vLLM 설치

vLLM은 CUDA 버전에 따라 설치 방법이 다릅니다. 공식 가이드를 참고하세요:
https://docs.vllm.ai/en/latest/getting_started/installation.html

### 2. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

## 기본 모델 설정

`app.py` 상단의 `MODEL_PATH`를 로컬에 저장된 Llama 모델 경로로 수정하세요:

```python
# app.py
MODEL_PATH = "/your/path/to/llama"   # ← 변경
혹은
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
```

## 실행

```bash
python app.py
```

서버가 시작되면 브라우저에서 `http://localhost:8000` 으로 접속합니다.

## 프로젝트 구조

```
korean_essay_rater/
├── app.py                            # FastAPI 서버 · vLLM 추론 · SSE 스트리밍
├── feature_extractor.py              # kiwipiepy 기반 형태소 분석 · 특징 추출
├── feature_inventory_14-1.json       # 사용 피처 목록 (141개)
├── essay_question_keyword_mapping.json  # 에세이 문항 · 핵심 키워드 매핑
├── requirements.txt
├── templates/
│   └── index.html                    # 단일 페이지 UI (Chart.js)
└── rater/                            # LoRA 어댑터
    ├── adapter_config.json           # LoRA 설정 (r=16, alpha=32, q/v_proj)
    ├── adapter_model.safetensors     # 어댑터 가중치
    └── ...                           # 토크나이저 파일
```

> **Git LFS 주의**: `adapter_model.safetensors`(~26 MB), `tokenizer.json`(~16 MB)는 용량이 크므로 Git LFS를 사용하거나 Hugging Face Hub에 별도 호스팅하는 것을 권장합니다.

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 UI |
| `GET` | `/api/questions` | 에세이 문항 목록 조회 |
| `POST` | `/api/analyze` | 에세이 채점 (SSE 스트리밍) |
| `POST` | `/api/text-analysis` | 형태소 분석 결과 조회 |

## LoRA 어댑터 정보

| 항목 | 값 |
|------|----|
| 기반 아키텍처 | Llama (Causal LM) |
| PEFT 방식 | LoRA |
| Rank (`r`) | 16 |
| Alpha | 32 |
| 적용 모듈 | `q_proj`, `v_proj` |
| Dropout | 0.05 |
