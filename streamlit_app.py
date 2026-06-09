"""Conversation Quality Score (CQS) — interactive Streamlit console.

An LLM-as-a-judge style evaluation demo for conversational AI. It scores
customer conversations on four rubric dimensions (relevance, helpfulness,
correctness, tone), rolls them into a single 0-100 CQS, and benchmarks the
automated judge against human labels.

This demo runs entirely on a deterministic, built-in heuristic judge — no API
key or setup required. The same heuristic powers a simple agent-reply simulator
in the "Score your own" tab so you can generate and grade a conversation
end to end.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from cqs_judge import DIMENSIONS, RUBRIC, cqs_from_scores, get_judge
from agent_sim import simulate_agent_reply

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
    initial_sidebar_state="collapsed",
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
        .agent-reply {{
            border:1px solid #e6e9e4; border-left:4px solid {COLORS["teal"]};
            border-radius:10px; padding:14px 18px; background:#ffffff;
            font-size:1rem; line-height:1.5;
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


@st.cache_resource
def get_mock_judge():
    return get_judge(mock=True)


def score_color(cqs: float) -> str:
    if cqs >= 80:
        return COLORS["teal"]
    if cqs >= 60:
        return COLORS["amber"]
    return COLORS["red"]


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
    r = pearson(j_all, h_all)
    cqs_mae = (df["judge CQS"] - df["human CQS"]).abs().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exact dimension match", f"{exact:.0f}%")
    c2.metric("Within ±1", f"{within1:.0f}%")
    c3.metric("CQS MAE (pts)", f"{cqs_mae:.1f}")
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
    st.bar_chart(dim_df, color=[COLORS["teal"], COLORS["amber"]])


# --------------------------------------------------------------------------- #
# Tab 2: Score your own  (two-step: simulate reply -> score it)
# --------------------------------------------------------------------------- #
def render_score_own(judge) -> None:
    st.markdown("#### Score your own conversation")
    st.caption(
        "Type a customer message, simulate the shopping assistant's reply, then "
        "score that reply on the four CQS dimensions."
    )

    default_msg = "I need lightweight running shoes under $80, size 10."
    user_text = st.text_area("Customer message", value=default_msg, height=120)

    col_sim, col_reset = st.columns([1, 1])
    with col_sim:
        if st.button("Simulate agent reply", type="primary"):
            st.session_state["sim_reply"] = simulate_agent_reply(user_text)
            st.session_state["sim_user"] = user_text
            # clear any earlier score when a new reply is generated
            st.session_state.pop("sim_judgement", None)
    with col_reset:
        if st.button("Clear"):
            for k in ("sim_reply", "sim_user", "sim_judgement"):
                st.session_state.pop(k, None)

    reply = st.session_state.get("sim_reply")
    if not reply:
        st.info("Enter a customer message and click **Simulate agent reply** to begin.")
        return

    st.markdown("##### Simulated agent reply")
    st.caption(
        "Generated by a built-in rule-based simulator (a stand-in for a live agent)."
    )
    st.markdown(f"<div class='agent-reply'>{reply}</div>", unsafe_allow_html=True)
    st.write("")

    # Step 2: the score option appears only once a reply exists.
    if st.button("Score this conversation", type="primary"):
        turns = [
            {"role": "user", "text": st.session_state.get("sim_user", user_text)},
            {"role": "assistant", "text": reply},
        ]
        st.session_state["sim_judgement"] = judge.score(turns)

    judgement = st.session_state.get("sim_judgement")
    if judgement is not None:
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
        "**Two-tier design.** In production, compliance-critical traffic gets human "
        "review while the long tail is scored cheaply by an LLM judge — and that "
        "automated signal is calibrated against human labels so you know how far to "
        "trust it. This demo uses a deterministic heuristic in place of the LLM so it "
        "runs with zero setup."
    )


def main() -> None:
    inject_styles()
    judge = get_mock_judge()

    st.markdown("<span class='cqs-pill'>LLM-as-a-judge</span>", unsafe_allow_html=True)
    st.title("Conversation Quality Score")
    st.markdown(
        f"<p style='color:{COLORS['muted']};font-size:1.05rem;margin-top:-6px'>"
        "Measure conversational-AI quality at scale, kept honest against human review."
        "</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Benchmark", "Score your own", "Rubric"])
    with tab1:
        render_benchmark(judge)
    with tab2:
        render_score_own(judge)
    with tab3:
        render_rubric()


if __name__ == "__main__":
    main()
