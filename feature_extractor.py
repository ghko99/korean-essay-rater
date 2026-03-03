"""Korean essay feature extractor using Kiwi + Bareun.

  1. Sentence splitting — Kiwi (more accurate Korean sentence segmentation)
  2. Morphological analysis — Bareun AnalyzeSyntax (per sentence)
  3. Grammar checking — Bareun CorrectError (spelling & spacing errors)
  4. v2 feature extraction — keywords, requirements, grammar for prompt

Requirements:
    pip install kiwipiepy bareunpy grpcio protobuf
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# ── Paths ────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent
_INVENTORY_PATH = _BASE_DIR / "feature_inventory_14-1.json"
FEATURE_INVENTORY: List[str] = json.loads(
    _INVENTORY_PATH.read_text(encoding="utf-8")
)["features"]

CIRCLED_NUMS = "①②③④⑤⑥⑦⑧"

# ── POS tag → feature name ───────────────────────────────────────────
POS_TO_FEATURE: Dict[str, str] = {
    "NNG":  "일반 명사",
    "NNP":  "고유 명사",
    "NNB":  "의존 명사",
    "NNM":  "단위를 나타내는 명사",
    "NNBC": "단위를 나타내는 명사",
    "NP":   "대명사",
    "NR":   "수사",
    "VV":   "동사",
    "VA":   "형용사",
    "VX":   "보조 용언",
    "VCP":  "긍정 지정사",
    "VCN":  "부정 지정사",
    "MM":   "관형사",
    "MMA":  "관형사",
    "MMD":  "관형사",
    "MMN":  "관형사",
    "MAG":  "일반 부사",
    "MAJ":  "접속 부사",
    "JKS":  "주격 조사",
    "JKC":  "보격 조사",
    "JKG":  "관형격 조사",
    "JKO":  "목적격 조사",
    "JKB":  "부사격 조사",
    "JKV":  "호격 조사",
    "JKQ":  "인용격 조사",
    "JX":   "보조사",
    "JC":   "접속 조사",
    "EP":   "선어말 어미",
    "EF":   "종결 어미",
    "EC":   "연결 어미",
    "ETN":  "명사형 전성 어미",
    "ETM":  "관형형 전성 어미",
    "XPN":  "체언 접두사",
    "XSN":  "명사 파생 접미사",
    "XSV":  "동사 파생 접미사",
    "XSA":  "형용사 파생 접미사",
    "XR":   "어근",
    "SL":   "외국어",
    "SH":   "한자",
    "SN":   "숫자",
    "IC":   "감탄사",
    "UNA":  "UNA",
}

OPEN_BRACKETS  = {"(", "[", "{", "<", "«", "〈", "《", "「", "『", "【", "〔", "\u201c", "\u2018"}
CLOSE_BRACKETS = {")", "]", "}", ">", "»", "〉", "》", "」", "』", "】", "〕", "\u201d", "\u2019"}
HYPHEN_TOKENS  = {"-", "‐", "‑", "‒", "–", "—", "―"}
COUNTER_TOKENS = {
    "개", "명", "대", "마리", "권", "장", "줄", "병", "잔", "번", "회", "채", "층", "살",
    "킬로그램", "kg", "km", "cm", "m", "g", "리터", "L",
}


# ── Token → feature label ───────────────────────────────────────────
def token_to_feature(token: str, pos: str) -> str:
    if pos == "SF":
        if token in {"...", "…", "⋯"}:
            return "줄임표"
        return "마침표, 물음표, 느낌표"
    if token in HYPHEN_TOKENS or pos == "SO":
        return "붙임표"
    if pos == "SSO" or token in OPEN_BRACKETS:
        return "여는 괄호"
    if pos == "SSC" or token in CLOSE_BRACKETS:
        return "닫는 괄호"
    if pos in {"SC", "SP", "SE", "SS"}:
        return "구분자"
    return POS_TO_FEATURE.get(pos, "")


# ── N-gram count ─────────────────────────────────────────────────────
def count_ngrams(sequence: List[str], pattern: List[str]) -> int:
    n = len(pattern)
    if n == 0 or len(sequence) < n:
        return 0
    return sum(1 for i in range(len(sequence) - n + 1) if sequence[i: i + n] == pattern)


# ── Feature extraction from raw token list ───────────────────────────
def extract_features_from_raw_tokens(
    raw_tokens: List[Dict[str, str]],
    sentence_count: int | None = None,
) -> Dict[str, int]:
    labels: List[str] = []
    counts: Counter = Counter()
    sent_end = 0

    token_pos_pairs = [
        (str(t.get("token", "")), str(t.get("tag", "")))
        for t in raw_tokens
    ]
    for i, (token, pos) in enumerate(token_pos_pairs):
        feat = token_to_feature(token, pos)
        if not feat and pos in {"NNB", "NNG"} and token in COUNTER_TOKENS and i > 0:
            if token_pos_pairs[i - 1][1] in {"SN", "NR"}:
                feat = "단위를 나타내는 명사"
        if not feat:
            continue
        labels.append(feat)
        counts[feat] += 1
        if feat == "마침표, 물음표, 느낌표":
            sent_end += 1

    out: Dict[str, int] = {f: 0 for f in FEATURE_INVENTORY}
    for f in FEATURE_INVENTORY:
        if f == "문장수":
            if sentence_count is not None:
                out[f] = sentence_count
            else:
                out[f] = sent_end if sent_end > 0 else (1 if labels else 0)
        elif "+" in f:
            out[f] = count_ngrams(labels, [x.strip() for x in f.split("+")])
        else:
            out[f] = counts.get(f, 0)
    return out


# ── Bareun gRPC clients ─────────────────────────────────────────────

class _BareunMorphClient:
    """Lazy-initialized Bareun AnalyzeSyntax gRPC client."""

    def __init__(self, api_key: str, ca_bundle_path: str):
        self._api_key = api_key
        self._ca_bundle_path = ca_bundle_path
        self._stub = None
        self._channel = None

    def _init_stub(self):
        if self._stub is not None:
            return
        import grpc
        from bareun import language_service_pb2_grpc as lg

        ca_cert = Path(self._ca_bundle_path).read_bytes()
        creds = grpc.ssl_channel_credentials(root_certificates=ca_cert)
        self._channel = grpc.secure_channel(
            "api.bareun.ai:443",
            creds,
            options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
        )
        self._stub = lg.LanguageServiceStub(self._channel)

    def analyze(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        """Run AnalyzeSyntax and return structured result.

        Returns:
            {
                "raw_tokens": [{"token": str, "tag": str}, ...],
                "sentences": [
                    {
                        "text": str,
                        "tokens": [
                            {
                                "surface": str,
                                "morphemes": [{"surface": str, "tag": str}, ...]
                            }, ...
                        ]
                    }, ...
                ]
            }
        """
        from bareun import language_service_pb2 as lp
        from bareun import lang_common_pb2 as lc

        self._init_stub()

        normalized = (text or "").strip()
        if not normalized:
            return {"raw_tokens": [], "sentences": []}

        doc = lc.Document()
        doc.content = normalized[:3000]
        doc.language = "ko_KR"
        req = lp.AnalyzeSyntaxRequest()
        req.document.CopyFrom(doc)
        req.encoding_type = lc.UTF32
        metadata = [("api-key", self._api_key)]

        for attempt in range(max_retries):
            try:
                response, _ = self._stub.AnalyzeSyntax.with_call(
                    request=req, metadata=metadata, timeout=30,
                )
                return self._parse_response(response)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    return {"raw_tokens": [], "sentences": [], "error": str(e)}

        return {"raw_tokens": [], "sentences": []}

    @staticmethod
    def _parse_response(response) -> Dict[str, Any]:
        from google.protobuf.json_format import MessageToDict
        result = MessageToDict(response)

        raw_tokens: List[Dict[str, str]] = []
        sentences: List[Dict[str, Any]] = []

        for sent_data in result.get("sentences", []):
            sent_text = sent_data.get("text", {}).get("content", "")
            sent_tokens = []

            for token_data in sent_data.get("tokens", []):
                surface = token_data.get("text", {}).get("content", "")
                morphemes = []
                for m in token_data.get("morphemes", []):
                    m_surface = m.get("text", {}).get("content", "")
                    m_tag = m.get("tag", "")
                    morphemes.append({"surface": m_surface, "tag": m_tag})
                    raw_tokens.append({"token": m_surface, "tag": m_tag})

                sent_tokens.append({"surface": surface, "morphemes": morphemes})

            sentences.append({"text": sent_text, "tokens": sent_tokens})

        return {"raw_tokens": raw_tokens, "sentences": sentences}


class _BareunGrammarClient:
    """Lazy-initialized Bareun CorrectError gRPC client."""

    def __init__(self, api_key: str, ca_bundle_path: str):
        self._api_key = api_key
        self._ca_bundle_path = ca_bundle_path
        self._stub = None
        self._channel = None

    def _init_stub(self):
        if self._stub is not None:
            return
        import grpc
        from bareun import revision_service_pb2_grpc as rs_grpc

        ca_cert = Path(self._ca_bundle_path).read_bytes()
        creds = grpc.ssl_channel_credentials(root_certificates=ca_cert)
        self._channel = grpc.secure_channel(
            "api.bareun.ai:443",
            creds,
            options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
        )
        self._stub = rs_grpc.RevisionServiceStub(self._channel)

    def check(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        from google.protobuf.json_format import MessageToDict
        from bareun import revision_service_pb2 as pb

        self._init_stub()

        request = pb.CorrectErrorRequest()
        request.document.content = text[:2000]
        request.document.language = "ko_KR"
        metadata = [("api-key", self._api_key)]

        for attempt in range(max_retries):
            try:
                response, _ = self._stub.CorrectError.with_call(
                    request=request, metadata=metadata, timeout=30,
                )
                result = MessageToDict(response)

                spelling_errors = []
                spacing_errors = []

                for block in result.get("revisedBlocks", []):
                    origin = block.get("origin", {}).get("content", "")
                    revised = block.get("revised", "")
                    if not origin or not revised or origin == revised:
                        continue
                    for rev in block.get("revisions", []):
                        category = rev.get("category", "")
                        error = {
                            "origin": origin.strip(),
                            "revised": rev.get("revised", revised).strip(),
                        }
                        if category == "SPACING":
                            spacing_errors.append(error)
                        elif category == "GRAMMER":
                            spelling_errors.append(error)

                return {
                    "spelling_errors": spelling_errors[:5],
                    "spacing_errors": spacing_errors[:5],
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    return {"spelling_errors": [], "spacing_errors": [], "error": str(e)}

        return {"spelling_errors": [], "spacing_errors": []}


# ── Keyword matching ─────────────────────────────────────────────────

def _parse_keyword_groups(kw_str: str) -> list[list[str]]:
    groups = []
    # Split by commas that are NOT inside parentheses
    parts = re.split(r",\s*(?![^(]*\))", kw_str)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Remove parentheses, treat contents as alternatives split by comma
        # e.g. "공기(기체, 대류현상)" -> ["공기", "기체", "대류현상"]
        alternatives = re.findall(r"[\w가-힣\-≠=+△□]+", part)
        if alternatives:
            groups.append(alternatives)
    return groups


def _match_keywords(essay: str, kw_str: str) -> tuple[list[str], int]:
    groups = _parse_keyword_groups(kw_str)
    essay_normalized = essay.lower().replace(" ", "")
    matched = []
    for group in groups:
        for alt in group:
            alt_clean = alt.lower().replace(" ", "")
            if alt_clean and alt_clean in essay_normalized:
                matched.append(group[0])
                break
    return matched, len(groups)


# ── Grammar result → string ──────────────────────────────────────────

def _grammar_result_to_str(grammar_result: Dict[str, Any]) -> str | None:
    if not grammar_result or "error" in grammar_result:
        return None

    sp_errs = grammar_result.get("spelling_errors", [])
    sc_errs = grammar_result.get("spacing_errors", [])

    grammar_str = f"맞춤법오류 {len(sp_errs)}건, 띄어쓰기오류 {len(sc_errs)}건"

    error_examples = []
    for e in sp_errs[:2]:
        error_examples.append(f"{e['origin']}→{e['revised']}")
    for e in sc_errs[:2]:
        error_examples.append(f"{e['origin']}→{e['revised']}")
    if error_examples:
        grammar_str += f" {error_examples}"

    return grammar_str


# ── Main FeatureExtractor ────────────────────────────────────────────

class FeatureExtractor:
    """Unified extractor: Kiwi sentence split + Bareun morphology + v2 prompt features.

    Produces:
      - Sentence splitting via Kiwi (more accurate for Korean)
      - Morphological analysis (per sentence) via Bareun AnalyzeSyntax
      - Feature counts (367 features from feature_inventory)
      - v2 prompt feature string (keywords, requirements, grammar)
    """

    def __init__(
        self,
        bareun_api_key: str,
        ca_bundle_path: str,
        question_requirements_path: str | None = None,
    ):
        from kiwipiepy import Kiwi
        self._kiwi = Kiwi()
        self._morph = _BareunMorphClient(bareun_api_key, ca_bundle_path)
        self._grammar = _BareunGrammarClient(bareun_api_key, ca_bundle_path)

        self._req_map: dict = {}
        if question_requirements_path:
            with open(question_requirements_path, encoding="utf-8") as f:
                self._req_map = json.load(f)

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using Kiwi."""
        sents = self._kiwi.split_into_sents(text)
        return [s.text.strip() for s in sents if s.text.strip()]

    def get_sentence_count(self, text: str) -> int:
        """Quick sentence count via Kiwi."""
        return len(self.split_sentences(text))

    def analyze_morphology(self, text: str) -> Dict[str, Any]:
        """Split sentences with Kiwi, then run Bareun on each sentence."""
        sentences_text = self.split_sentences(text)
        all_raw_tokens: List[Dict[str, str]] = []
        sentences: List[Dict[str, Any]] = []

        for sent_text in sentences_text:
            result = self._morph.analyze(sent_text)
            sent_tokens = []
            for sent_data in result.get("sentences", []):
                for token_data in sent_data.get("tokens", []):
                    sent_tokens.append(token_data)
            raw_tokens = result.get("raw_tokens", [])
            all_raw_tokens.extend(raw_tokens)
            sentences.append({"text": sent_text, "tokens": sent_tokens})

        return {"raw_tokens": all_raw_tokens, "sentences": sentences}

    def extract_full(
        self,
        question: str,
        keywords: str,
        essay: str,
    ) -> Dict[str, Any]:
        """Full extraction: morphology + features + v2 prompt string.

        Returns:
            {
                "morph": { bareun morphology result },
                "feature_counts": { feature: count, ... },
                "prompt_features": "v2 feature string for prompt",
                "grammar_result": { spelling/spacing errors },
            }
        """
        # 1. Kiwi sentence split + Bareun morphological analysis per sentence
        morph = self.analyze_morphology(essay)
        raw_tokens = morph.get("raw_tokens", [])
        sent_count = len(morph.get("sentences", []))

        # 2. Feature counts (367 features)
        feature_counts = extract_features_from_raw_tokens(
            raw_tokens, sentence_count=sent_count,
        )

        # 3. Grammar check
        grammar_result = self._grammar.check(essay)

        # 4. Build v2 prompt feature string
        prompt_features = self._build_prompt_features(
            question, keywords, essay, grammar_result,
        )

        return {
            "morph": morph,
            "feature_counts": feature_counts,
            "prompt_features": prompt_features,
            "grammar_result": grammar_result,
        }

    def _build_prompt_features(
        self,
        question: str,
        keywords: str,
        essay: str,
        grammar_result: Dict[str, Any],
    ) -> str:
        lines = []

        # 핵심 키워드 — 괄호 대체어를 "A/B/C" 형태로 정리
        groups = _parse_keyword_groups(keywords)
        formatted_kw = ", ".join("/".join(g) for g in groups)
        lines.append(f"- 핵심 키워드: {formatted_kw}")

        # 에세이에 포함된 키워드
        matched, total = _match_keywords(essay, keywords)
        if matched:
            matched_str = ", ".join(matched)
            lines.append(f"- 에세이에 포함된 키워드: {matched_str} ({len(matched)}/{total}개)")
        else:
            lines.append(f"- 에세이에 포함된 키워드: 없음 (0/{total}개)")

        # 질문 요구사항
        if question in self._req_map:
            reqs = self._req_map[question]["requirements"]
            req_items = " ".join(
                CIRCLED_NUMS[i] + r for i, r in enumerate(reqs) if i < len(CIRCLED_NUMS)
            )
            lines.append(f"- 질문 요구사항: {req_items}")

        # 어법 검사
        grammar_str = _grammar_result_to_str(grammar_result)
        if grammar_str:
            lines.append(f"- 어법 검사: {grammar_str}")

        return "\n".join(lines)
