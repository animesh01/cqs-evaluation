"""LLM-as-a-judge for Chat Quality Score (CQS).

Two modes:
  - real:  uses the Anthropic API (set ANTHROPIC_API_KEY, `pip install anthropic`)
  - mock:  a deterministic, text-only heuristic that needs no API key, so the
           demo runs out of the box. It is intentionally weak: it cannot verify
           facts or catch subtle irrelevance, which is exactly why a real LLM
           (or a human) is used as the judge in production.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

RUBRIC = {
    "relevance": "Does the assistant directly address what the customer asked?",
    "helpfulness": "Does it move the customer toward their goal (specific, actionable)?",
    "correctness": "Is the information accurate and free of fabrication?",
    "tone": "Is it polite, clear, and appropriately concise?",
}
DIMENSIONS = list(RUBRIC.keys())


def conversation_to_text(turns) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in turns)


def cqs_from_scores(scores) -> float:
    """Mean of the 1-5 dimension scores, scaled to 0-100."""
    vals = [scores[d] for d in DIMENSIONS]
    return round(sum(vals) / len(vals) / 5 * 100, 1)


@dataclass
class Judgement:
    relevance: int
    helpfulness: int
    correctness: int
    tone: int
    rationale: str = ""
    dim_reasons: dict | None = None

    def scores(self) -> dict:
        return {d: getattr(self, d) for d in DIMENSIONS}

    def reasons(self) -> dict:
        return self.dim_reasons or {d: "" for d in DIMENSIONS}

    def cqs(self) -> float:
        return cqs_from_scores(self.scores())


# --------------------------------------------------------------------------- #
# Real judge: a frontier LLM scores each conversation against the rubric.
# --------------------------------------------------------------------------- #
_PROMPT = """You are an evaluation judge scoring one customer conversation with an AI shopping assistant.

Score each dimension from 1 (poor) to 5 (excellent):
{rubric}

Conversation:
{conversation}

Return ONLY a JSON object (no prose, no code fences) with integer scores and a one-sentence rationale:
{{"relevance": int, "helpfulness": int, "correctness": int, "tone": int, "rationale": "..."}}"""


def build_prompt(turns) -> str:
    rubric = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    return _PROMPT.format(rubric=rubric, conversation=conversation_to_text(turns))


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0) if match else text)


class LLMJudge:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        from anthropic import Anthropic  # lazy import so mock mode needs no install

        self.model = model
        # If api_key is given (e.g. pasted into the Streamlit UI) use it; otherwise
        # fall back to the ANTHROPIC_API_KEY environment variable.
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def score(self, turns) -> Judgement:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": build_prompt(turns)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        data = _parse_json(text)
        return Judgement(
            **{d: int(data[d]) for d in DIMENSIONS},
            rationale=str(data.get("rationale", "")),
        )


# --------------------------------------------------------------------------- #
# Mock judge: deterministic heuristic baseline (reads conversation text only,
# never the human labels). Useful for running the pipeline with zero setup.
# --------------------------------------------------------------------------- #
_REFUSAL = ("not able", "can't help", "cannot help", "unable to", "i can't")
_HEDGE = ("depends", "many ", "various", "could look", "different brands")
_POLITE = ("great", "happy to", "would you like", "want me to", "let me", "sure", "whenever you are")
_WORD = re.compile(r"[a-z0-9$]+")


def _content_words(s: str) -> set:
    return {w for w in _WORD.findall(s.lower()) if len(w) > 3}


class MockJudge:
    """Deterministic, text-only heuristic. A weak baseline by design."""

    model = "mock-heuristic"

    def score(self, turns) -> Judgement:
        user = " ".join(t["text"] for t in turns if t["role"] == "user")
        asst = " ".join(t["text"] for t in turns if t["role"] == "assistant")
        low = asst.lower()

        uw, aw = _content_words(user), _content_words(asst)
        overlap = len(uw & aw) / max(1, len(uw))
        relevance = 5 if overlap >= 0.40 else 4 if overlap >= 0.25 else 3 if overlap >= 0.12 else 2
        rel_reason = (
            "Closely matches what the customer asked about."
            if relevance >= 4
            else "Only partly matches what the customer asked." if relevance == 3
            else "Doesn't really match what was asked."
        )

        specifics = len(re.findall(r"\$?\d", asst)) + (1 if "?" in asst else 0)
        helpfulness = 5 if specifics >= 4 else 4 if specifics >= 2 else 3 if specifics >= 1 else 2
        help_reason = (
            "Gives specific options and a clear next step."
            if helpfulness >= 4
            else "Gives some detail but could be more specific." if helpfulness == 3
            else "Too vague to really help."
        )

        refused = any(k in low for k in _REFUSAL)
        if refused:
            helpfulness = 1
            relevance = min(relevance, 2)
            help_reason = "Turns the request down, so it doesn't help."
            rel_reason = "Declines the request instead of addressing it."

        hedged = any(k in low for k in _HEDGE)
        correctness = 3 if hedged else 4  # heuristic cannot verify facts
        cor_reason = (
            "Vague, non-committal wording — hard to call it accurate."
            if hedged
            else "Nothing looks obviously wrong (note: this check can't verify facts)."
        )

        polite = any(k in low for k in _POLITE)
        tone = 5 if polite else 4
        terse = len(asst.split()) < 8
        if terse:
            tone = min(tone, 2)
            tone_reason = "Very short and abrupt."
        else:
            tone_reason = (
                "Warm and polite."
                if polite
                else "Neutral, businesslike tone."
            )

        dim_reasons = {
            "relevance": rel_reason,
            "helpfulness": help_reason,
            "correctness": cor_reason,
            "tone": tone_reason,
        }

        return Judgement(
            relevance, helpfulness, correctness, tone,
            "Automated score based on a simple rule-based check (no LLM).",
            dim_reasons=dim_reasons,
        )


def get_judge(mock: bool = False, model: str = "claude-sonnet-4-6", api_key: str | None = None):
    return MockJudge() if mock else LLMJudge(model=model, api_key=api_key)
