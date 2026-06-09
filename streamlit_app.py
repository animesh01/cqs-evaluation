"""Conversation Quality Score (CQS) — evaluation console for an AI shopping assistant.

Scores customer conversations on four rubric dimensions (relevance, helpfulness,
correctness, tone), rolls them into a single 0-100 CQS, and benchmarks an
automated judge against human labels — with per-dimension reasoning from both.

Runs entirely on a deterministic, built-in heuristic judge — no API key needed.
"""
from __future__ import annotations

import html
import json
import math
import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cqs_judge import DIMENSIONS, RUBRIC, cqs_from_scores, get_judge
from agent_sim import simulate_agent_reply

ROOT = Path(__file__).resolve().parent
POOL_DATA = ROOT / "data" / "conversation_pool.json"

TEAL = "#0d766e"
INK = "#12221f"
MUTED = "#62706d"
MINT = "#dff4ef"
SAND = "#f4efe7"
AMBER = "#b66a16"
RED = "#b84040"

METRIC_HELP = {
    "relevance": "Does the assistant directly address what the customer asked? "
    "Off-topic or generic replies score low.",
    "helpfulness": "Does the reply move the customer toward their goal with specific, "
    "actionable detail (named products, prices, next steps)?",
    "correctness": "Is the information accurate and free of fabrication or unsupported "
    "guarantees? Ignoring a stated constraint (budget, size, diet) lowers this.",
    "tone": "Is the reply polite, clear, and appropriately concise — neither terse nor pushy?",
}
DIM_ICON = {"relevance": "🎯", "helpfulness": "🧭", "correctness": "✅", "tone": "💬"}

st.set_page_config(page_title="CQS — Conversation Quality Score", page_icon="✦",
                   layout="wide", initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------- #
# Styles + hero
# --------------------------------------------------------------------------- #
def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family:"DM Sans",sans-serif; color:{INK}; }}
        h1,h2,h3,h4 {{ font-family:"Manrope",sans-serif !important; letter-spacing:-0.02em; }}
        .hero {{
            border-radius:20px; padding:30px 34px; margin-bottom:18px; color:#fff;
            background:linear-gradient(135deg,#0d766e 0%,#0f8a73 55%,#14a989 100%);
            position:relative; overflow:hidden;
        }}
        .hero h1 {{ color:#fff !important; font-size:2.05rem; margin:6px 0 6px; }}
        .hero p {{ color:#e7f6f1; font-size:1.02rem; line-height:1.55; max-width:80%; margin:0; }}
        .hero .pill {{ display:inline-block; padding:4px 13px; border-radius:999px;
            background:rgba(255,255,255,0.18); color:#fff; font-size:0.74rem; font-weight:700;
            letter-spacing:0.07em; text-transform:uppercase; }}
        .hero-art {{ position:absolute; right:-10px; top:-10px; opacity:0.9; }}
        .stat-row {{ display:flex; gap:14px; margin-top:18px; flex-wrap:wrap; }}
        .stat {{ background:rgba(255,255,255,0.14); border-radius:12px; padding:10px 16px; }}
        .stat .n {{ font-size:1.35rem; font-weight:800; font-family:Manrope; }}
        .stat .l {{ font-size:0.74rem; text-transform:uppercase; letter-spacing:0.05em; opacity:0.85; }}
        .rubric-box {{ border:1px solid #e6e9e4; border-radius:14px; padding:14px 18px; background:#fff; }}
        .rubric-row {{ display:flex; gap:10px; padding:5px 0; font-size:0.92rem; align-items:baseline; }}
        .rubric-name {{ min-width:118px; font-weight:700; color:{TEAL}; text-transform:capitalize; }}
        .conv-msg {{ border-radius:10px; padding:10px 14px; margin:4px 0; font-size:0.93rem; line-height:1.5; }}
        .conv-user {{ background:{MINT}; }}
        .conv-asst {{ background:#f5f6f4; border-left:3px solid {TEAL}; border-radius:0 10px 10px 0; }}
        .agent-reply {{ border:1px solid #e6e9e4; border-left:4px solid {TEAL};
            border-radius:0 10px 10px 0; padding:14px 18px; background:#fff; font-size:1rem; line-height:1.5; }}
        .cqs-card {{ border:1px solid #e6e9e4; border-radius:14px; padding:18px 20px; background:#fff; }}
        .cmptbl {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
        .cmptbl th {{ color:{MUTED}; text-align:left; padding:6px 10px; font-weight:700;
            border-bottom:1px solid #eceee9; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.03em; }}
        .cmptbl td {{ padding:9px 10px; border-bottom:1px solid #f1f2ef; vertical-align:top; }}
        .scorechip {{ display:inline-block; min-width:30px; text-align:center; padding:3px 8px;
            border-radius:8px; font-weight:700; font-size:0.88rem; }}
        .reason {{ color:{MUTED}; font-size:0.85rem; line-height:1.4; }}
        .mathline {{ background:{SAND}; border-radius:10px; padding:10px 14px; font-size:0.9rem;
            display:flex; gap:22px; flex-wrap:wrap; margin:6px 0 12px; }}
        .mathline b {{ font-family:Manrope; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    art = (
        "<svg class='hero-art' width='220' height='160' viewBox='0 0 220 160' fill='none'>"
        "<circle cx='150' cy='70' r='95' fill='rgba(255,255,255,0.06)'/>"
        "<circle cx='185' cy='30' r='55' fill='rgba(255,255,255,0.05)'/>"
        # speech bubbles
        "<rect x='40' y='44' width='86' height='40' rx='12' fill='rgba(255,255,255,0.92)'/>"
        "<rect x='52' y='56' width='52' height='6' rx='3' fill='#0d766e'/>"
        "<rect x='52' y='68' width='36' height='6' rx='3' fill='#9fe1cb'/>"
        "<polygon points='56,84 56,98 72,84' fill='rgba(255,255,255,0.92)'/>"
        "<rect x='96' y='92' width='86' height='40' rx='12' fill='rgba(255,255,255,0.22)'/>"
        "<rect x='108' y='104' width='52' height='6' rx='3' fill='#fff'/>"
        "<rect x='108' y='116' width='36' height='6' rx='3' fill='#cfeee3'/>"
        # check gauge
        "<circle cx='168' cy='66' r='22' fill='none' stroke='#fff' stroke-width='5' "
        "stroke-dasharray='104 138' stroke-linecap='round' transform='rotate(-90 168 66)'/>"
        "<path d='M159 66 l6 6 l12 -13' stroke='#fff' stroke-width='4' fill='none' "
        "stroke-linecap='round' stroke-linejoin='round'/>"
        "</svg>"
    )
    st.markdown(
        f"""
        <div class="hero">
          {art}
          <span class="pill">LLM-as-a-judge · evaluation harness</span>
          <h1>Conversation Quality Score</h1>
          <p>Grades every conversation from a customer-facing <b>AI shopping assistant</b> on
          four quality dimensions, rolls them into a single <b>0–100 CQS</b>, and keeps the
          automated judge honest against human reviewers.</p>
          <div class="stat-row">
            <div class="stat"><div class="n">4</div><div class="l">dimensions</div></div>
            <div class="stat"><div class="n">0–100</div><div class="l">CQS scale</div></div>
            <div class="stat"><div class="n">22</div><div class="l">labeled convos</div></div>
            <div class="stat"><div class="n">judge + human</div><div class="l">side by side</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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
def load_pool() -> list[dict]:
    with open(POOL_DATA) as f:
        return json.load(f)


@st.cache_resource
def get_mock_judge():
    return get_judge(mock=True)


def score_color(cqs: float) -> str:
    return TEAL if cqs >= 80 else AMBER if cqs >= 60 else RED


def chip(value: int) -> str:
    if value >= 4:
        bg, fg = "#e1f5ee", TEAL
    elif value >= 3:
        bg, fg = "#faeeda", AMBER
    else:
        bg, fg = "#fcebeb", RED
    return f"<span class='scorechip' style='background:{bg};color:{fg}'>{value}</span>"


def esc(s: str) -> str:
    return html.escape(s or "")


# --------------------------------------------------------------------------- #
# Rubric box
# --------------------------------------------------------------------------- #
def render_rubric_box() -> None:
    rows = "".join(
        f"<div class='rubric-row'><span class='rubric-name'>{DIM_ICON[k]} {k}</span>"
        f"<span style='color:{MUTED}'>{v}</span></div>"
        for k, v in RUBRIC.items()
    )
    st.markdown(
        f"<div class='rubric-box'>"
        f"<div style='font-weight:700;margin-bottom:4px;font-family:Manrope'>The CQS rubric</div>"
        f"<div style='font-size:0.85rem;color:{MUTED};margin-bottom:8px'>"
        f"Each turn is scored 1–5 on four dimensions; the average is rescaled to a 0–100 CQS "
        f"(e.g. an average of 3/5 → 60, 4/5 → 80).</div>{rows}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def score_pool(judge, convs):
    rows, j_all, h_all = [], [], []
    for c in convs:
        jd = judge.score(c["turns"])
        js, jr = jd.scores(), jd.reasons()
        hs = c["human_label"]
        hr = c.get("human_reason", {d: "" for d in DIMENSIONS})
        for d in DIMENSIONS:
            j_all.append(js[d])
            h_all.append(hs[d])
        rows.append({"conv": c, "js": js, "jr": jr, "hs": hs, "hr": hr,
                     "j_cqs": cqs_from_scores(js), "h_cqs": cqs_from_scores(hs),
                     "j_avg": sum(js.values()) / 4, "h_avg": sum(hs.values()) / 4})
    return rows, j_all, h_all


def metrics_header(j_all, h_all):
    n = len(j_all)
    exact = sum(1 for a, b in zip(j_all, h_all) if a == b) / n * 100
    within1 = sum(1 for a, b in zip(j_all, h_all) if abs(a - b) <= 1) / n * 100
    r = pearson(j_all, h_all)
    cqs_mae = sum(abs(a - b) for a, b in zip(j_all, h_all)) / n / 5 * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exact match", f"{exact:.0f}%",
              help="Share of dimension grades where judge and human gave the identical 1–5 score.")
    c2.metric("Within ±1", f"{within1:.0f}%",
              help="Share of grades where judge and human are within one point — the practical agreement bar.")
    c3.metric("CQS MAE", f"{cqs_mae:.1f} pts",
              help="Mean absolute error between judge and human on the 0–100 CQS scale.")
    c4.metric("Correlation r", f"{r:.2f}",
              help="Pearson correlation between judge and human grades across all dimensions.")


def agreement_scatter_svg(rows) -> str:
    """Pure-SVG scatter of judge CQS (x) vs human CQS (y) with an agreement diagonal."""
    W = H = 360
    pad = 46
    plot = W - 2 * pad

    def px(v):  # 0-100 -> x pixel
        return pad + v / 100 * plot

    def py(v):  # 0-100 -> y pixel (inverted)
        return pad + plot - v / 100 * plot

    # gridlines + axis ticks at 0,25,50,75,100
    grid = ""
    for t in (0, 25, 50, 75, 100):
        grid += (f"<line x1='{px(t):.1f}' y1='{pad}' x2='{px(t):.1f}' y2='{pad+plot}' "
                 f"stroke='#eef0ec' stroke-width='1'/>")
        grid += (f"<line x1='{pad}' y1='{py(t):.1f}' x2='{pad+plot}' y2='{py(t):.1f}' "
                 f"stroke='#eef0ec' stroke-width='1'/>")
        grid += (f"<text x='{px(t):.1f}' y='{pad+plot+16}' font-size='10' fill='{MUTED}' "
                 f"text-anchor='middle'>{t}</text>")
        grid += (f"<text x='{pad-8}' y='{py(t)+3:.1f}' font-size='10' fill='{MUTED}' "
                 f"text-anchor='end'>{t}</text>")

    # perfect-agreement diagonal
    diag = (f"<line x1='{px(0):.1f}' y1='{py(0):.1f}' x2='{px(100):.1f}' y2='{py(100):.1f}' "
            f"stroke='{TEAL}' stroke-width='1.5' stroke-dasharray='5 4'/>")

    # points, colored by gap size
    pts = ""
    for r in rows:
        x, y = px(r["j_cqs"]), py(r["h_cqs"])
        gap = abs(r["j_cqs"] - r["h_cqs"])
        fill = TEAL if gap < 5 else AMBER if gap < 15 else RED
        pts += (f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' fill='{fill}' fill-opacity='0.75' "
                f"stroke='#fff' stroke-width='1.5'><title>{r['conv']['id']}: "
                f"judge {r['j_cqs']:.0f} vs human {r['h_cqs']:.0f}</title></circle>")

    labels = (f"<text x='{pad+plot/2:.1f}' y='{H-6}' font-size='11' fill='{MUTED}' "
              f"text-anchor='middle'>LLM judge CQS →</text>"
              f"<text x='14' y='{pad+plot/2:.1f}' font-size='11' fill='{MUTED}' "
              f"text-anchor='middle' transform='rotate(-90 14 {pad+plot/2:.1f})'>← Human CQS</text>"
              f"<text x='{px(78):.1f}' y='{py(86):.1f}' font-size='10' fill='{TEAL}'>perfect agreement</text>")

    return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:380px' "
            f"xmlns='http://www.w3.org/2000/svg' role='img'>"
            f"<rect x='{pad}' y='{pad}' width='{plot}' height='{plot}' fill='#fff' "
            f"stroke='#e6e9e4'/>{grid}{diag}{pts}{labels}</svg>")


def dimension_drift(rows):
    """Mean signed judge-minus-human per dimension, to show where the judge drifts."""
    out = []
    for d in DIMENSIONS:
        diffs = [r["js"][d] - r["hs"][d] for r in rows]
        mean = sum(diffs) / len(diffs)
        out.append((d, mean))
    return out


def agreement_scatter_fig(rows, focused_id=None):
    """Interactive Plotly scatter: judge CQS (x) vs human CQS (y).

    Returns a figure whose points carry the conversation id in customdata, so a
    click can be mapped back to the conversation for drill-down.
    """
    fig = go.Figure()
    # perfect-agreement diagonal
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines",
        line=dict(color=TEAL, width=1.5, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    xs = [r["j_cqs"] for r in rows]
    ys = [r["h_cqs"] for r in rows]
    ids = [r["conv"]["id"] for r in rows]
    domains = [r["conv"]["domain"].replace("_", " ") for r in rows]
    colors, sizes, lines = [], [], []
    for r in rows:
        gap = abs(r["j_cqs"] - r["h_cqs"])
        colors.append(TEAL if gap < 5 else AMBER if gap < 15 else RED)
        focused = focused_id == r["conv"]["id"]
        sizes.append(20 if focused else 13)
        lines.append(3 if focused else 1.5)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors,
                    line=dict(color="#ffffff", width=lines)),
        customdata=list(zip(ids, domains)),
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                      "judge %{x:.0f} · human %{y:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", size=12, color=MUTED),
        xaxis=dict(title="LLM judge CQS →", range=[-3, 103], gridcolor="#eef0ec",
                   zeroline=False, dtick=25),
        yaxis=dict(title="← Human CQS", range=[-3, 103], gridcolor="#eef0ec",
                   zeroline=False, dtick=25),
        dragmode=False,
    )
    return fig


def render_benchmark(judge) -> None:
    st.markdown("#### Benchmark — judge vs human labels")
    st.caption("A sample of human-labeled shopping conversations, scored live by the automated "
               "judge. Draw a fresh sample and watch the agreement shift.")

    pool = load_pool()
    sample_size = min(8, len(pool))
    top = st.columns([1, 1, 2])
    with top[0]:
        if st.button("🔀 Sample new set", type="primary"):
            st.session_state["bench_seed"] = random.randint(0, 10_000)
            st.session_state.pop("focus_id", None)
    with top[1]:
        if st.button("Show full pool"):
            st.session_state["bench_seed"] = "ALL"

    seed = st.session_state.get("bench_seed", 42)
    if seed == "ALL":
        convs = pool
        st.caption(f"Showing all {len(pool)} labeled conversations.")
    else:
        convs = random.Random(seed).sample(pool, sample_size)
        st.caption(f"Showing a sample of {sample_size} of {len(pool)} labeled conversations.")

    rows, j_all, h_all = score_pool(judge, convs)
    metrics_header(j_all, h_all)

    # Agreement scatter + per-dimension drift, side by side
    st.write("")
    viz_l, viz_r = st.columns([1, 1])
    with viz_l:
        st.markdown("##### Judge vs human")
        st.caption("Each point is one conversation. Click a point to open that conversation "
                   "below; distance from the dashed line is the disagreement.")
        focused = st.session_state.get("focus_id")
        fig = agreement_scatter_fig(rows, focused_id=focused)
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                                key="scatter", selection_mode="points")
        # map a click back to a conversation id
        try:
            pts = event["selection"]["points"]
            if pts:
                cd = pts[-1].get("customdata")
                if cd:
                    st.session_state["focus_id"] = cd[0]
        except (KeyError, TypeError, IndexError):
            pass
        if st.session_state.get("focus_id"):
            fc = st.session_state["focus_id"]
            cc1, cc2 = st.columns([3, 1])
            cc1.caption(f"Focused: **{fc}** — its card is open below.")
            if cc2.button("Clear focus"):
                st.session_state.pop("focus_id", None)
                st.rerun()
    with viz_r:
        st.markdown("##### Where the judge drifts")
        st.caption("Average judge-minus-human gap per dimension. Positive = the judge is too "
                   "generous; negative = too harsh.")
        for d, mean in dimension_drift(rows):
            direction = "over-rates" if mean > 0.05 else "under-rates" if mean < -0.05 else "matches"
            bar_col = RED if abs(mean) >= 0.5 else AMBER if abs(mean) >= 0.2 else TEAL
            pct = min(abs(mean) / 2 * 100, 100)
            st.markdown(
                f"<div style='margin:6px 0'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.88rem'>"
                f"<span style='text-transform:capitalize'>{DIM_ICON[d]} {d}</span>"
                f"<span style='color:{MUTED}'>{mean:+.2f} ({direction})</span></div>"
                f"<div style='background:#eef0ec;border-radius:5px;height:8px;margin-top:3px'>"
                f"<div style='width:{pct:.0f}%;background:{bar_col};height:8px;border-radius:5px'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("##### Conversations")
    sort_col = st.columns([1.4, 2])
    with sort_col[0]:
        order = st.selectbox(
            "Sort by",
            ["Biggest disagreement first", "Conversation id", "Lowest human CQS first"],
            label_visibility="collapsed",
        )
    st.caption("Click any conversation to see the exchange, the CQS math, and the per-dimension "
               "scores with reasoning from both the LLM judge and the human reviewer.")

    if order == "Biggest disagreement first":
        rows = sorted(rows, key=lambda r: abs(r["j_cqs"] - r["h_cqs"]), reverse=True)
    elif order == "Lowest human CQS first":
        rows = sorted(rows, key=lambda r: r["h_cqs"])
    else:
        rows = sorted(rows, key=lambda r: r["conv"]["id"])

    # a clicked point pulls its conversation to the top and opens it
    focus_id = st.session_state.get("focus_id")
    if focus_id:
        rows = sorted(rows, key=lambda r: r["conv"]["id"] != focus_id)

    for row in rows:
        c = row["conv"]
        user_text = next((t["text"] for t in c["turns"] if t["role"] == "user"), "")
        asst_text = next((t["text"] for t in c["turns"] if t["role"] == "assistant"), "")
        gap = row["j_cqs"] - row["h_cqs"]
        gap_str = f"{gap:+.0f}" if abs(gap) >= 0.5 else "match"
        is_focus = c["id"] == focus_id
        star = "⭐ " if is_focus else ""
        header = (f"{star}{c['id']}  ·  {c['domain'].replace('_', ' ')}   —   "
                  f"judge {row['j_cqs']:.0f}  vs  human {row['h_cqs']:.0f}   ({gap_str})")
        with st.expander(header, expanded=is_focus):
            st.markdown(
                f"<div class='conv-msg conv-user'><strong>Customer:</strong> {esc(user_text)}</div>"
                f"<div class='conv-msg conv-asst'><strong>Agent:</strong> {esc(asst_text)}</div>",
                unsafe_allow_html=True,
            )
            # explicit CQS math for both
            st.markdown(
                f"<div class='mathline'>"
                f"<span>🤖 <b>LLM:</b> avg {row['j_avg']:.2f}/5 → "
                f"<b style='color:{score_color(row['j_cqs'])}'>{row['j_cqs']:.1f} CQS</b></span>"
                f"<span>🧑 <b>Human:</b> avg {row['h_avg']:.2f}/5 → "
                f"<b style='color:{score_color(row['h_cqs'])}'>{row['h_cqs']:.1f} CQS</b></span>"
                f"<span style='color:{MUTED}'>Δ {gap:+.1f} CQS</span></div>",
                unsafe_allow_html=True,
            )
            # comparison table: score + reasoning for each side
            t = "<table class='cmptbl'><tr><th>Dimension</th><th>LLM</th><th>LLM reasoning</th>"
            t += "<th>Human</th><th>Human reasoning</th></tr>"
            for d in DIMENSIONS:
                t += (
                    f"<tr><td style='text-transform:capitalize;font-weight:600'>{DIM_ICON[d]} {d}</td>"
                    f"<td>{chip(row['js'][d])}</td>"
                    f"<td class='reason'>{esc(row['jr'].get(d, ''))}</td>"
                    f"<td>{chip(row['hs'][d])}</td>"
                    f"<td class='reason'>{esc(row['hr'].get(d, ''))}</td></tr>"
                )
            t += "</table>"
            st.markdown(t, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Score your own
# --------------------------------------------------------------------------- #
def render_score_own(judge) -> None:
    st.markdown("#### Score your own conversation")
    st.caption("Type a customer message, simulate the shopping assistant's reply, then score it.")

    default_msg = "I need lightweight running shoes under $80, size 10."
    user_text = st.text_area("Customer message", value=default_msg, height=110)

    col_sim, col_reset = st.columns([1, 1])
    with col_sim:
        if st.button("Simulate agent reply", type="primary"):
            st.session_state["sim_reply"] = simulate_agent_reply(user_text)
            st.session_state["sim_user"] = user_text
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
    st.caption("Generated by a built-in rule-based simulator (a stand-in for a live agent).")
    st.markdown(f"<div class='agent-reply'>{esc(reply)}</div>", unsafe_allow_html=True)
    st.write("")

    if st.button("Score this conversation", type="primary"):
        turns = [{"role": "user", "text": st.session_state.get("sim_user", user_text)},
                 {"role": "assistant", "text": reply}]
        st.session_state["sim_judgement"] = judge.score(turns)

    jd = st.session_state.get("sim_judgement")
    if jd is not None:
        cqs = jd.cqs()
        avg = sum(jd.scores().values()) / 4
        st.markdown(
            f"<div class='cqs-card'><span style='color:{MUTED};font-size:0.85rem'>"
            f"avg {avg:.2f}/5 → </span>"
            f"<h2 style='display:inline;margin:0;color:{score_color(cqs)}'>{cqs:.1f}</h2>"
            f"<span style='font-size:1rem;color:{MUTED}'> / 100 CQS</span></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        scores, reasons = jd.scores(), jd.reasons()
        for d in DIMENSIONS:
            cols = st.columns([1, 4])
            cols[0].metric(f"{DIM_ICON[d]} {d.capitalize()}", f"{scores[d]}/5", help=METRIC_HELP[d])
            cols[1].markdown(
                f"<div style='padding-top:14px' class='reason'>{esc(reasons[d])}</div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_styles()
    judge = get_mock_judge()
    hero()
    render_rubric_box()
    st.write("")
    tab1, tab2 = st.tabs(["Benchmark", "Score your own"])
    with tab1:
        render_benchmark(judge)
    with tab2:
        render_score_own(judge)


if __name__ == "__main__":
    main()
