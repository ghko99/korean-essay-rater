import asyncio
import math
import os
import uuid
import json
import re

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind

from feature_extractor import FeatureExtractor, token_to_feature

# ── Paths & config ────────────────────────────────────────────────────
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
ADAPTER_DIR = str(BASE_DIR / "rater")

BAREUN_API_KEY = os.environ["BAREUN_API_KEY"]
CA_BUNDLE_PATH = str(BASE_DIR / "artifacts" / "bareun_ca_bundle.pem")
QUESTION_REQ_PATH = str(BASE_DIR / "artifacts" / "question_requirements.json")

# ── Feature extractor (bareun-based) ─────────────────────────────────
extractor = FeatureExtractor(
    bareun_api_key=BAREUN_API_KEY,
    ca_bundle_path=CA_BUNDLE_PATH,
    question_requirements_path=QUESTION_REQ_PATH,
)

# ── Essay question / keyword mapping ─────────────────────────────────
_MAPPING_PATH = BASE_DIR / "essay_question_keyword_mapping.json"
question_keyword_map: dict = json.loads(_MAPPING_PATH.read_text(encoding="utf-8"))
questions: list[str] = list(question_keyword_map.keys())

# ── Prompt template (v2 features) ────────────────────────────────────
PROMPT_FORMAT = """
### 지시문:
너는 '채점 기준 (루브릭)'에 따라 학생의 에세이를 평가하는 AI 채점기다.
### 에세이 질문:
{question}
### 학생 에세이:
 {essay}
### 관련 정보:
{features}
### 채점 기준 (루브릭):
- 과제 수행의 충실성:
  - 9점 (최고): 지시문의 요구와 조건에 맞게 과제를 매우 완벽하게 수행
  - 1점 (최저): 지시문의 요구와 조건을 고려하지 못하고 과제수행도 매우 미흡함
- 설명의 명료성:
  - 9점 (최고): 설명하고자 하는 대상에 초점을 맞추어 매우 완벽하게 서술
  - 1점 (최저): 설명하고자 하는 대상에 초점을 맞운 서술이 매우 미흡함
- 설명의 구체성:
  - 9점 (최고): 구체적인 세부 내용을 제시하여 대상을 매우 잘 설명함
  - 1점 (최저): 설명하는 대상에 대해 세부 내용을 제시하지 않음
- 설명의 적절성:
  - 9점 (최고): 지시문의 요구에 맞게 내용을 매우 잘 구성(핵심어 4개이상)
  - 1점 (최저): 지시문의 요구 맞는 내용구성이 잘 안됨(핵심어 없음)
- 문장의 연결성:
  - 9점 (최고): 문장이 오류없이 잘 구성되고, 문장간 연결이 매우 잘 구성됨
  - 1점 (최저): 문장이 오류없이 잘 구성되고, 문장간 연결이 매우 부족함
- 글의 통일성:
  - 9점 (최고): 지시문의 요구에 맞게 통일성을 매우 잘 갖춰 구성됨
  - 1점 (최저): 지시문의 요구 조건을 고려하지 못하고 통일성이 매우 미흡하게 구성됨
- 어휘의 적절성:
  - 9점 (최고): 어휘 선택이 탁월하고 문장이 어법에 맞으며 수려하다.
  - 1점 (최저): 어휘 선택이 적절 하지 않고 문장표현이 어색하다.
- 어법의 적절성:
  - 9점 (최고): 모든 경우 어법에 맞는 맞춤법과 띄어쓰기를 사용한다.
  - 1점 (최저): 맞춤법과 띄어쓰기 사용이 전반적으로 미흡하다.

먼저 8개의 점수를 공백으로 구분해 한 줄에 출력하고, 빈 줄 이후에 각 항목에 대한 상세 피드백을 작성하라.
"""

# ── vLLM stop tokens ──────────────────────────────────────────────────
_tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
_stop_ids: list[int] = []
if getattr(_tok, "eos_token_id", None) is not None:
    _stop_ids.append(_tok.eos_token_id)
for _t in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
    try:
        _tid = _tok.convert_tokens_to_ids(_t)
        if isinstance(_tid, int) and _tid != _tok.unk_token_id:
            _stop_ids.append(_tid)
    except Exception:
        pass
_stop_ids = sorted(set(_stop_ids))
print("STOP IDS:", _stop_ids)

_DIGIT_TOKEN_IDS: dict[int, int] = {}
for _d in range(1, 10):
    for _form in [str(_d), f" {_d}"]:
        _ids = _tok.encode(_form, add_special_tokens=False)
        if len(_ids) == 1 and _ids[0] not in _DIGIT_TOKEN_IDS:
            _DIGIT_TOKEN_IDS[_ids[0]] = _d
print(f"DIGIT TOKEN IDS: {_DIGIT_TOKEN_IDS}")


def build_sampling_params(n: int = 1) -> SamplingParams:
    return SamplingParams(
        n=n,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        stop_token_ids=_stop_ids,
        ignore_eos=False,
        output_kind=RequestOutputKind.DELTA,
    )


async def _get_score_probs(
    engine: AsyncLLMEngine, prompt: str, lora_req: LoRARequest
) -> list[list[float]] | None:
    params = SamplingParams(
        n=1,
        max_tokens=25,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        logprobs=10,
        stop_token_ids=_stop_ids,
        ignore_eos=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    request_id = str(uuid.uuid4())
    output = None
    try:
        async for out in engine.generate(
            prompt=prompt,
            sampling_params=params,
            request_id=request_id,
            lora_request=lora_req,
        ):
            output = out
    except Exception:
        return None

    if not output or not output.outputs:
        return None

    completion = output.outputs[0]
    if not completion.logprobs or not completion.token_ids:
        return None

    score_dists: list[list[float]] = []

    for tid, lp_dict in zip(completion.token_ids, completion.logprobs):
        if lp_dict is None:
            continue
        if tid not in _DIGIT_TOKEN_IDS:
            continue

        score_lps: dict[int, float] = {}
        for ltid, lp in lp_dict.items():
            if ltid in _DIGIT_TOKEN_IDS:
                d = _DIGIT_TOKEN_IDS[ltid]
                if d not in score_lps or lp.logprob > score_lps[d]:
                    score_lps[d] = lp.logprob

        if not score_lps:
            continue

        max_lp = max(score_lps.values())
        raw = {s: math.exp(lp - max_lp) for s, lp in score_lps.items()}
        total = sum(raw.values())
        score_dists.append([round(raw.get(s, 0.0) / total, 4) for s in range(1, 10)])

        if len(score_dists) == 8:
            break

    return score_dists if len(score_dists) == 8 else None


# ── App ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading vLLM engine...")
    await get_engine()
    print("[STARTUP] vLLM engine ready.")
    yield


app = FastAPI(title="Korean Essay Rater", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── vLLM Engine (lazy singleton) ──────────────────────────────────────
_engine: AsyncLLMEngine | None = None
_engine_lock = asyncio.Lock()


async def get_engine() -> AsyncLLMEngine:
    global _engine
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                engine_args = AsyncEngineArgs(
                    model=MODEL_PATH,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.8,
                    max_model_len=4096,
                    disable_log_stats=True,
                    enable_lora=True,
                    max_loras=1,
                    max_lora_rank=16,
                )
                _engine = AsyncLLMEngine.from_engine_args(engine_args)
    return _engine


LORA_REQ = LoRARequest("aes_adapter", 1, ADAPTER_DIR)

CRITERIA = [
    "과제 수행의 충실성",
    "설명의 명료성",
    "설명의 구체성",
    "설명의 적절성",
    "문장의 연결성",
    "글의 통일성",
    "어휘의 적절성",
    "어법의 적절성",
]

# ── CJK filter ────────────────────────────────────────────────────────
_CJK_RE = re.compile(
    r'[\u3040-\u309f'
    r'\u30a0-\u30ff'
    r'\u4e00-\u9fff'
    r'\u3400-\u4dbf'
    r'\uff66-\uff9f]+'
)


def _filter_cjk(text: str) -> str:
    text = _CJK_RE.sub("", text)
    return re.sub(r" {2,}", " ", text)


# ── Request schemas ───────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    question_idx: int
    essay: str


class EssayTextRequest(BaseModel):
    essay: str


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/questions")
async def get_questions():
    return [
        {"idx": i, "question": q, "keywords": question_keyword_map[q]}
        for i, q in enumerate(questions)
    ]


@app.post("/api/sentence-count")
async def sentence_count(req: EssayTextRequest):
    """문장수만 체크 (bareun AnalyzeSyntax)."""
    cnt = await asyncio.to_thread(extractor.get_sentence_count, req.essay)
    return {"sentence_count": cnt}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if req.question_idx < 0 or req.question_idx >= len(questions):
        return {"error": "잘못된 질문 번호입니다."}

    question = questions[req.question_idx]
    keywords = question_keyword_map[question]

    async def event_stream():
        # Pre-check: sentence count via bareun
        sent_count = await asyncio.to_thread(extractor.get_sentence_count, req.essay)
        if sent_count < 2:
            yield _sse({
                "type": "error",
                "message": "2문장 이상일 때만 평가할 수 있습니다.",
                "sentence_count": sent_count,
            })
            yield _sse({"type": "done"})
            return

        # Step 1: full feature extraction (morphology + v2 features + grammar)
        yield _sse({"type": "status", "message": "언어 특징 분석 중..."})
        extraction = await asyncio.to_thread(
            extractor.extract_full, question, keywords, req.essay,
        )

        morph = extraction["morph"]
        feature_counts = extraction["feature_counts"]
        prompt_features = extraction["prompt_features"]
        grammar_result = extraction["grammar_result"]

        # Send features to frontend
        nonzero_features = {k: v for k, v in feature_counts.items() if v > 0}

        # Build morpheme data for report display
        morph_display = []
        for sent in morph.get("sentences", []):
            sent_tokens = []
            for tok in sent.get("tokens", []):
                morphemes = []
                for m in tok.get("morphemes", []):
                    feat = token_to_feature(m["surface"], m["tag"])
                    morphemes.append({
                        "surface": m["surface"],
                        "tag": m["tag"],
                        "feature": feat or m["tag"],
                    })
                sent_tokens.append({"surface": tok["surface"], "morphemes": morphemes})
            morph_display.append({"text": sent["text"], "words": sent_tokens})

        yield _sse({
            "type": "features",
            "prompt_features": prompt_features,
            "feature_counts": nonzero_features,
            "morph_sentences": morph_display,
            "grammar": {
                "spelling_errors": grammar_result.get("spelling_errors", []),
                "spacing_errors": grammar_result.get("spacing_errors", []),
            },
        })

        # Step 2: build prompt & generate
        prompt = PROMPT_FORMAT.format(
            question=question,
            essay=req.essay,
            features=prompt_features,
        )

        n_samples = 50
        yield _sse({"type": "status", "message": "AI 분석 중..."})

        engine = await get_engine()
        sampling_params = build_sampling_params(n=n_samples)
        request_id = str(uuid.uuid4())

        logprobs_task = asyncio.create_task(
            _get_score_probs(engine, prompt, LORA_REQ)
        )

        full_texts = [""] * n_samples
        async for out in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=LORA_REQ,
        ):
            for completion in out.outputs:
                idx = completion.index
                filtered = _filter_cjk(completion.text)
                full_texts[idx] += filtered
                if idx == 0 and filtered:
                    yield _sse({"type": "token", "text": filtered})
            if out.finished:
                break

        # Step 3: aggregate & select
        avg_scores, valid_responses, distributions = _aggregate_scores(full_texts)
        evaluators = _select_evaluators(valid_responses, avg_scores)

        yield _sse({
            "type": "scores",
            "criteria": CRITERIA,
            "scores": avg_scores,
            "distributions": distributions,
            "valid_count": len(valid_responses),
            "total_count": n_samples,
        })
        yield _sse({
            "type": "evaluators",
            "evaluators": [
                {
                    "label": f"평가자 {i + 1}",
                    "scores": scores,
                    "rubrics": _parse_rubric_feedback(feedback),
                }
                for i, (scores, feedback) in enumerate(evaluators)
            ],
            "valid_count": len(valid_responses),
            "total_count": n_samples,
        })

        score_probs = await logprobs_task
        if score_probs:
            yield _sse({"type": "score_probs", "probs": score_probs})

        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Helpers ───────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_scores(text: str) -> list[int] | None:
    lines = text.strip().split("\n")
    if not lines:
        return None
    nums = re.findall(r"\d+", lines[0].strip())
    if len(nums) < 8:
        return None
    scores = [int(n) for n in nums[:8]]
    if not all(1 <= s <= 9 for s in scores):
        return None
    return scores


def _extract_feedback(text: str) -> str:
    idx = text.find("\n\n")
    if idx >= 0:
        return text[idx + 2:].strip()
    idx2 = text.find("\n")
    if idx2 >= 0 and re.match(r"^\d[\d\s]+$", text[:idx2]):
        return text[idx2 + 1:].strip()
    return text.strip()


def _aggregate_scores(texts: list[str]) -> tuple[list[int], list[tuple[list[int], str]], list[list[int]]]:
    valid: list[tuple[list[int], str]] = []
    for t in texts:
        scores = _parse_scores(t)
        if scores is not None:
            valid.append((scores, _extract_feedback(t)))
    if not valid:
        return [0] * 8, [], [[] for _ in range(8)]
    avg = [round(sum(v[0][i] for v in valid) / len(valid)) for i in range(8)]
    distributions = [[v[0][i] for v in valid] for i in range(8)]
    return avg, valid, distributions


def _select_evaluators(
    valid_responses: list[tuple[list[int], str]],
    avg_scores: list[int],
    max_k: int = 5,
) -> list[tuple[list[int], str]]:
    if len(valid_responses) <= max_k:
        return list(valid_responses)
    scored = sorted(
        valid_responses,
        key=lambda x: sum(abs(s - a) for s, a in zip(x[0], avg_scores)),
    )
    return scored[:max_k]


def _parse_rubric_feedback(feedback_text: str) -> list[str]:
    sections: dict[str, str] = {}
    pattern = re.compile(r"[-•*]\s*(.+?)\s*[:：]\s*\n")
    matches = list(pattern.finditer(feedback_text))
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(feedback_text)
        body = re.sub(r"^\s*[\"']|[\"']\s*$", "", feedback_text[start:end].strip()).strip()
        for criterion in CRITERIA:
            if criterion in name or name in criterion:
                sections[criterion] = body
                break
    return [sections.get(c, "") for c in CRITERIA]


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
