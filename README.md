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
- **형태소 분석**: Kiwi(문장 분리) + Bareun(형태소 분석, 맞춤법/띄어쓰기 검사) 기반 언어적 특징 추출
- **SSE 스트리밍**: 채점 과정(특징 분석 → 토큰 생성 → 최종 점수)을 실시간 스트림으로 전달

## 시스템 구성

```
사용자 브라우저
    │  SSE / REST API
    ▼
FastAPI (app.py)
    ├── Kiwi               문장 분리
    ├── Bareun gRPC API    형태소 분석 · 맞춤법/띄어쓰기 검사
    └── vLLM AsyncEngine   Llama + LoRA 어댑터 추론
```

## 요구 사항

- Python 3.10+
- CUDA 지원 GPU (vLLM 실행에 필요)
- Docker (Bareun 형태소 분석기 실행에 필요)
- base model : meta-llama/Llama-3.1-8B-Instruct (별도 다운로드 필요)

## 설치

### 1. vLLM 설치

vLLM은 CUDA 버전에 따라 설치 방법이 다릅니다. 공식 가이드를 참고하세요:
https://docs.vllm.ai/en/latest/getting_started/installation.html

### 2. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Bareun 형태소 분석기 설정

Bareun은 형태소 분석(Tagger)과 맞춤법 교정(Corrector)을 제공합니다.
형태소 분석은 로컬 Docker 컨테이너에서, 맞춤법 교정은 클라우드 API(`api.bareun.ai:443`)를 사용합니다.

#### 3-1. Docker 컨테이너 설치 (최초 1회)

```bash
# Docker 이미지 다운로드
docker pull bareunai/bareun:latest

# 볼륨 디렉토리 생성
mkdir -p ~/bareun/var

# 컨테이너 생성 및 실행
docker run -d --restart unless-stopped --name bareun \
  -p 5656:5656 -p 9902:9902 \
  -v ~/bareun/var:/bareun/var \
  bareunai/bareun:latest

# API 키 등록
docker exec bareun /bareun/bin/bareun -reg <YOUR_BAREUN_API_KEY>
```

> `--restart unless-stopped` 옵션으로 Docker Desktop이 켜지면 자동으로 시작됩니다.
> 컨테이너가 안 떠있으면 `docker start bareun`으로 시작하세요.

#### 3-2. CA 번들 생성 (맞춤법 교정 API용)

맞춤법 교정 API는 `api.bareun.ai:443`에 gRPC TLS로 접속하며, SSL 중간 인증서가 필요합니다.

```bash
# Sectigo 중간 인증서 다운로드
curl -s "http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt" \
  -o /tmp/sectigo_inter.der

# DER → PEM 변환
openssl x509 -in /tmp/sectigo_inter.der -inform DER -outform PEM \
  -o /tmp/sectigo_intermediate.pem

# certifi CA + 중간 인증서 합쳐서 CA 번들 생성
cat /tmp/sectigo_intermediate.pem \
    $(python -c "import certifi; print(certifi.where())") \
    > artifacts/bareun_ca_bundle.pem
```

> 인증서는 2027-03-22에 만료됩니다. 만료 후 위 절차를 다시 수행하세요.

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 항목을 설정하세요:

```env
BAREUN_API_KEY=your-bareun-api-key
MODEL_PATH=/your/path/to/llama          # 생략 시 meta-llama/Llama-3.1-8B-Instruct 사용
```

- `BAREUN_API_KEY`: [Bareun](https://bareun.ai/) API 키 (필수)
- `MODEL_PATH`: Llama 모델 경로 또는 Hugging Face 모델 ID (선택)

## 실행

```bash
python app.py
```

서버가 시작되면 브라우저에서 `http://localhost:8000` 으로 접속합니다.

## 프로젝트 구조

```
korean_essay_rater/
├── app.py                              # FastAPI 서버 · vLLM 추론 · SSE 스트리밍
├── feature_extractor.py                # Kiwi + Bareun 기반 형태소 분석 · 특징 추출
├── feature_inventory_14-1.json         # 사용 피처 목록 (141개)
├── essay_question_keyword_mapping.json # 에세이 문항 · 핵심 키워드 매핑
├── requirements.txt
├── .env                                # 환경 변수 (BAREUN_API_KEY 등, git 제외)
├── artifacts/
│   ├── bareun_ca_bundle.pem            # Bareun gRPC TLS 인증서 (git 제외)
│   └── question_requirements.json      # 질문별 요구사항 정의
├── templates/
│   └── index.html                      # 단일 페이지 UI (Chart.js)
└── rater/                              # LoRA 어댑터
    ├── adapter_config.json             # LoRA 설정 (r=16, alpha=32, q/v_proj)
    ├── adapter_model.safetensors       # 어댑터 가중치
    └── ...                             # 토크나이저 파일
```

> **Git LFS 주의**: `adapter_model.safetensors`(~26 MB), `tokenizer.json`(~16 MB)는 용량이 크므로 Git LFS를 사용하거나 Hugging Face Hub에 별도 호스팅하는 것을 권장합니다.

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 UI |
| `GET` | `/api/questions` | 에세이 문항 목록 조회 |
| `POST` | `/api/analyze` | 에세이 채점 (SSE 스트리밍) |
| `POST` | `/api/sentence-count` | 문장 수 체크 |

## LoRA 어댑터 정보

| 항목 | 값 |
|------|----|
| 기반 아키텍처 | Llama (Causal LM) |
| PEFT 방식 | LoRA |
| Rank (`r`) | 16 |
| Alpha | 32 |
| 적용 모듈 | `q_proj`, `v_proj` |
| Dropout | 0.05 |
