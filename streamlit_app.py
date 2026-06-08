"""Conversation Quality Score (CQS) — interactive Streamlit console.

An LLM-as-a-judge evaluation demo for conversational AI. Scores customer
conversations on four rubric dimensions (relevance, helpfulness, correctness,
tone), rolls them into a single 0-100 CQS, and calibrates the automated judge
against human labels.

Runs out of the box with a deterministic MOCK judge (no API key). Paste an
Anthropic API key in the sidebar to switch to the real LLM judge.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from cqs_judge import DIMENSIONS, RUBRIC, cqs_from_scores, get_judge

ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = ROOT / "data" / "sample_conversations.json"

COLORS = {
    "ink": "#12221f",
    "muted": "#62706d",
    "teal": "#0d766e",
    "mint": "#dff4ef",
    "sand": "#f4efe7",
    "amber": "#b66a16",
    "red": "#b84040",
}

st.set_page_config(
    page_title="CQS — Conversation Quality Score",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: "DM Sans", sans-serif; color: {COLORS["ink"]}; }}
        h1, h2, h3 {{ font-family: "Manrope", sans-serif !important; letter-spacing: -0.03em; }}
        .cqs-pill {{
            display:inline-block; padding:2px 10px; border-radius:999px;
            background:{COLORS["mint"]}; color:{COLORS["teal"]};
            font-size:0.8rem; font-weight:600; margin-bottom:8px;
        }}
        .cqs-card {{
            border:1px solid #e6e9e4; border-radius:14px; padding:18px 20px;
            background:#ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else float("nan")


@st.cache_data
def load_conversations() -> list[dict]:
    with open(SAMPLE_DATA) as f:
        return json.load(f)


def score_color(cqs: float) -> str:
    if cqs >= 80:
        return COLORS["teal"]
    if cqs >= 60:
        return COLORS["amber"]
    return COLORS["red"]


# --------------------------------------------------------------------------- #
# Sidebar: judge configuration
# --------------------------------------------------------------------------- #
def sidebar_config() -> tuple[bool, str, str | None]:
    st.sidebar.markdown("### Judge configuration")
    mode = st.sidebar.radio(
        "Judge mode",
        ["Mock (no API key, free)", "Real LLM judge (Anthropic)"],
        help="Mock is a deterministic keyword heuristic that runs instantly. "
        "The real judge calls the Anthropic API and is far more accurate.",
    )
    use_mock = mode.startswith("Mock")

    api_key = None
    model = "claude-sonnet-4-6"
    if not use_mock:
        # Pull a pre-set key from Streamlit secrets if the owner configured one,
        # otherwise let the visitor paste their own. The text box value lives only
        # in this session and is never logged or written to disk.
        secret_key = None
        try:
            secret_key = st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            secret_key = None

        api_key = st.sidebar.text_input(
            "Anthropic API key",
            value="",
            type="password",
            help="Your key is used only for this session. Never commit a key to the repo.",
        ) or secret_key

        model = st.sidebar.text_input("Model", value="claude-sonnet-4-6")
        if not api_key:
            st.sidebar.warning("Enter an API key, or switch to Mock mode to run for free.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<span style='color:{COLORS['muted']};font-size:0.85rem'>"
        "CQS = mean of four 1–5 rubric scores, rescaled to 0–100.</span>",
        unsafe_allow_html=True,
    )
    return use_mock, model, api_key


def judge_or_warn(use_mock: bool, model: str, api_key: str | None):
    if use_mock:
        return get_judge(mock=True)
    if not api_key:
        return None
    try:
        return get_judge(mock=False, model=model, api_key=api_key)
    except Exception as exc:  # missing anthropic package, bad key construction, etc.
        st.error(f"Could not initialise the LLM judge: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Tab 1: Benchmark
# --------------------------------------------------------------------------- #
def render_benchmark(judge) -> None:
    st.markdown("#### Benchmark — judge vs human labels")
    st.caption(
        "Scores a fixed, human-labeled set of conversations and reports how well "
        "the automated judge agrees with human reviewers."
    )

    convs = load_conversations()
    if st.button("Run benchmark", type="primary"):
        rows, j_all, h_all = [], [], []
        progress = st.progress(0.0, text="Scoring conversations…")
        for i, c in enumerate(convs):
            judgement = judge.score(c["turns"])
            js = judgement.scores()
            hs = c["human_label"]
            j_cqs = cqs_from_scores(js)
            h_cqs = cqs_from_scores(hs)
            for d in DIMENSIONS:
                j_all.append(js[d])
                h_all.append(hs[d])
            rows.append(
                {
                    "id": c["id"],
                    "domain": c.get("domain", ""),
                    "judge CQS": j_cqs,
                    "human CQS": h_cqs,
                    "gap": round(j_cqs - h_cqs, 1),
                    **{f"j_{d}": js[d] for d in DIMENSIONS},
                    **{f"h_{d}": hs[d] for d in DIMENSIONS},
                    "rationale": judgement.rationale,
                }
            )
            progress.progress((i + 1) / len(convs), text=f"Scored {i + 1}/{len(convs)}")
        progress.empty()
        st.session_state["bench_rows"] = rows
        st.session_state["bench_j"] = j_all
        st.session_state["bench_h"] = h_all

    if "bench_rows" not in st.session_state:
        st.info("Click **Run benchmark** to score the labeled set.")
        return

    rows = st.session_state["bench_rows"]
    j_all = st.session_state["bench_j"]
    h_all = st.session_state["bench_h"]
    df = pd.DataFrame(rows)

    n = len(j_all)
    exact = sum(1 for a, b in zip(j_all, h_all) if a == b) / n * 100
    within1 = sum(1 for a, b in zip(j_all, h_all) if abs(a - b) <= 1) / n * 100
    mae = sum(abs(a - b) for a, b in zip(j_all, h_all)) / n
    r = pearson(j_all, h_all)
    psr_mae = (df["judge CQS"] - df["human CQS"]).abs().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exact dimension match", f"{exact:.0f}%")
    c2.metric("Within ±1", f"{within1:.0f}%")
    c3.metric("CQS MAE (pts)", f"{psr_mae:.1f}")
    c4.metric("Correlation r", f"{r:.2f}")

    st.markdown("##### Per-conversation scores")
    show = df[["id", "domain", "judge CQS", "human CQS", "gap", "rationale"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("##### Per-dimension judge-vs-human (mean over set)")
    dim_df = pd.DataFrame(
        {
            "dimension": DIMENSIONS,
            "judge": [df[f"j_{d}"].mean() for d in DIMENSIONS],
            "human": [df[f"h_{d}"].mean() for d in DIMENSIONS],
        }
    ).set_index("dimension")
    st.bar_chart(dim_df)


# --------------------------------------------------------------------------- #
# Tab 2: Score your own
# --------------------------------------------------------------------------- #
def render_score_own(judge) -> None:
    st.markdown("#### Score your own conversation")
    st.caption("Paste a short customer ↔ assistant exchange and let the judge grade it.")

    default_user = "I need lightweight running shoes under $80, size 10."
    default_asst = (
        "Here are three lightweight running shoes under $80 in size 10: the Avia Flow "
        "($59), the Athletic Works Glide ($45), and the No Boundaries Lite ($72). Want "
        "me to add one to your cart?"
    )
    col_u, col_a = st.columns(2)
    user_text = col_u.text_area("Customer message", value=default_user, height=140)
    asst_text = col_a.text_area("Assistant reply", value=default_asst, height=140)

    if st.button("Score this conversation", type="primary"):
        turns = [
            {"role": "user", "text": user_text},
            {"role": "assistant", "text": asst_text},
        ]
        judgement = judge.score(turns)
        cqs = judgement.cqs()
        st.markdown(
            f"<div class='cqs-card'><span class='cqs-pill'>CQS</span>"
            f"<h2 style='margin:0;color:{score_color(cqs)}'>{cqs:.1f}<span "
            f"style='font-size:1rem;color:{COLORS['muted']}'> / 100</span></h2></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        scores = judgement.scores()
        cols = st.columns(len(DIMENSIONS))
        for col, d in zip(cols, DIMENSIONS):
            col.metric(d.capitalize(), f"{scores[d]}/5")
        if judgement.rationale:
            st.caption(f"Rationale: {judgement.rationale}")


# --------------------------------------------------------------------------- #
# Tab 3: Rubric
# --------------------------------------------------------------------------- #
def render_rubric() -> None:
    st.markdown("#### The CQS rubric")
    st.caption("Each conversation is scored 1–5 on four dimensions; the mean is rescaled to 0–100.")
    rub = pd.DataFrame(
        [{"dimension": k.capitalize(), "what it measures": v} for k, v in RUBRIC.items()]
    )
    st.table(rub)
    st.markdown(
        "**Two-tier design.** Compliance-critical traffic gets 100% human review; "
        "the long tail is scored cheaply by the LLM judge — and that automated signal "
        "is calibrated against the human labels so you know how far to trust it."
    )


def main() -> None:
    inject_styles()
    st.markdown("<span class='cqs-pill'>LLM-as-a-judge</span>", unsafe_allow_html=True)
    st.title("Conversation Quality Score")
    st.markdown(
        f"<p style='color:{COLORS['muted']};font-size:1.05rem;margin-top:-6px'>"
        "Measure conversational-AI quality at scale, kept honest against human review."
        "</p>",
        unsafe_allow_html=True,
    )

    use_mock, model, api_key = sidebar_config()
    judge = judge_or_warn(use_mock, model, api_key)

    tab1, tab2, tab3 = st.tabs(["Benchmark", "Score your own", "Rubric"])
    with tab1:
        if judge is None:
            st.info("Configure the judge in the sidebar to run the benchmark.")
        else:
            render_benchmark(judge)
    with tab2:
        if judge is None:
            st.info("Configure the judge in the sidebar to score a conversation.")
        else:
            render_score_own(judge)
    with tab3:
        render_rubric()


if __name__ == "__main__":
    main()
