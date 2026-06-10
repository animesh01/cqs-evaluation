"""Chat Quality Score (CQS) — evaluation console for an AI shopping assistant.

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


def _find_pool() -> Path:
    """Locate conversation_pool.json whether it sits in data/ or the repo root."""
    for candidate in (ROOT / "data" / "conversation_pool.json",
                      ROOT / "conversation_pool.json"):
        if candidate.exists():
            return candidate
    # default (will raise a clear error if truly missing)
    return ROOT / "data" / "conversation_pool.json"


POOL_DATA = _find_pool()

# Dark theme palette
BG = "#ecebf4"          # cool light canvas, faint violet undertone (CQS variant)
SURFACE = "#ffffff"     # raised card surface
SURFACE2 = "#f2f1fb"    # secondary surface / chips bg (violet tint)
BORDER = "#e6e4f2"      # hairline border
TEAL = "#6257d6"        # primary accent — indigo-violet (distinguishes CQS from PRQ blue)
TEAL_DEEP = "#5046c4"   # deeper indigo
INK = "#1a1830"         # near-black text (violet-ink)
MUTED = "#7b7790"       # muted text
MINT = "#eef0fb"        # customer-bubble bg (light indigo tint)
SAND = "#f1f3fb"        # math-line bg (light tint)
AMBER = "#dd9421"       # warning signal
RED = "#e8654f"         # negative signal
CHARCOAL = "#1d1a33"    # deep focal base
GREEN = "#22b892"       # positive signal

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

st.set_page_config(page_title="CQS — Chat Quality Score", page_icon="✦",
                   layout="wide", initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------- #
# Styles + hero
# --------------------------------------------------------------------------- #
def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        /* cool light canvas with faint violet top-corner light (CQS identity) */
        .stApp {{ background:
            radial-gradient(1200px 500px at 18% -8%, #f6f4fc 0%, rgba(246,244,252,0) 60%),
            linear-gradient(180deg, #ecebf4 0%, #e4e2f0 100%); }}
        html, body, [class*="css"] {{ font-family:"DM Sans",sans-serif; color:{INK}; }}
        h1,h2,h3,h4,h5,h6 {{ font-family:"Manrope",sans-serif !important; letter-spacing:-0.025em; color:{INK} !important; }}
        p, span, div, label, li, td, th {{ color:{INK}; }}
        .stCaption, [data-testid="stCaptionContainer"] {{ color:{MUTED} !important; }}

        /* ---- hero: CQS keeps a rich INDIGO gradient (distinct from PRQ's white hero) ---- */
        .hero {{ position:relative; overflow:hidden; border-radius:24px; padding:32px 36px; margin-bottom:20px; color:#fff;
            background:linear-gradient(135deg,#4a3fb0 0%,#6257d6 55%,#8a7ef0 100%);
            box-shadow:0 2px 2px rgba(40,32,90,.10), 0 16px 32px -10px rgba(60,48,160,.4), 0 44px 70px -24px rgba(60,48,160,.45); }}
        .hero::before {{ content:""; position:absolute; inset:0 0 auto 0; height:1px; border-radius:24px 24px 0 0;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent); }}
        .hero h1 {{ color:#fff !important; font-size:2.15rem; margin:10px 0 8px; }}
        .hero p {{ color:#ece9fb !important; font-size:1.02rem; line-height:1.55; max-width:80%; margin:0; }}
        .hero p b {{ color:#fff !important; }}
        .hero .pill {{ display:inline-block; padding:6px 14px; border-radius:999px;
            background:rgba(255,255,255,0.22); color:#fff !important; font-size:0.72rem; font-weight:700;
            letter-spacing:0.07em; text-transform:uppercase; box-shadow:inset 0 1px 0 rgba(255,255,255,.3); }}
        .hero-art {{ position:absolute; right:24px; top:18px; opacity:0.96;
            filter:drop-shadow(0 12px 18px rgba(40,32,90,.3)); }}
        .stat-row {{ display:flex; gap:14px; margin-top:20px; flex-wrap:wrap; }}
        .stat {{ background:rgba(255,255,255,0.16); border-radius:13px; padding:11px 17px;
            box-shadow:inset 0 1px 0 rgba(255,255,255,.18); }}
        .stat .n {{ font-size:1.35rem; font-weight:800; font-family:Manrope; color:#fff; }}
        .stat .l {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; opacity:0.9; color:#fff; }}

        /* shared raised-card recipe for light surfaces */
        .rubric-box, .cqs-card {{ position:relative; background:linear-gradient(180deg,#fff,#fafaff);
            border:1px solid {BORDER}; border-radius:18px; padding:18px 20px;
            box-shadow:0 1px 1px rgba(40,32,90,.04), 0 6px 12px -3px rgba(40,32,90,.10), 0 20px 34px -12px rgba(40,32,90,.16); }}
        .rubric-box::before, .cqs-card::before {{ content:""; position:absolute; inset:0 0 auto 0; height:1px; border-radius:18px 18px 0 0;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,.9),transparent); }}
        .rubric-row {{ display:flex; gap:10px; padding:5px 0; font-size:0.92rem; align-items:baseline; }}
        .rubric-name {{ min-width:118px; font-weight:700; color:{TEAL_DEEP}; text-transform:capitalize; }}

        /* conversation bubbles — light tints */
        .conv-msg {{ border-radius:12px; padding:10px 14px; margin:5px 0; font-size:0.93rem; line-height:1.5; }}
        .conv-user {{ background:{MINT}; border:1px solid #e2e0f6; }}
        .conv-asst {{ background:#fff; border:1px solid {BORDER}; border-left:3px solid {TEAL};
            border-radius:0 12px 12px 0; box-shadow:0 4px 10px -8px rgba(40,32,90,.3); }}
        .agent-reply {{ border:1px solid {BORDER}; border-left:4px solid {TEAL}; border-radius:0 12px 12px 0;
            padding:14px 18px; background:#fff; font-size:1rem; line-height:1.5;
            box-shadow:0 6px 14px -10px rgba(40,32,90,.3); }}
        .cmptbl {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
        .cmptbl th {{ color:{MUTED}; text-align:left; padding:7px 10px; font-weight:700;
            border-bottom:1px solid {BORDER}; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.03em;
            background:{SURFACE2}; }}
        .cmptbl td {{ padding:9px 10px; border-bottom:1px solid {BORDER}; vertical-align:top; color:{INK}; }}
        .scorechip {{ display:inline-block; min-width:30px; text-align:center; padding:3px 8px;
            border-radius:8px; font-weight:700; font-size:0.88rem; box-shadow:inset 0 1px 0 #fff; }}
        .reason {{ color:{MUTED}; font-size:0.85rem; line-height:1.4; }}
        .mathline {{ background:{SAND}; border:1px solid {BORDER}; border-radius:12px; padding:11px 15px; font-size:0.9rem;
            display:flex; gap:22px; flex-wrap:wrap; margin:6px 0 12px; }}
        .mathline b {{ font-family:Manrope; color:{INK}; }}

        /* ---- expanders: header matches an indigo-tint pill, dark readable text ---- */
        [data-testid="stExpander"] {{ border:1px solid {BORDER} !important; border-radius:14px; background:#fff;
            box-shadow:0 6px 14px -10px rgba(40,32,90,.3); overflow:hidden; }}
        [data-testid="stExpander"] details > summary,
        [data-testid="stExpander"] summary {{
            background:#f0effb !important; border-bottom:1px solid #e2e0f6 !important; }}
        [data-testid="stExpander"] details > summary,
        [data-testid="stExpander"] details > summary *,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] [class*="Header"],
        [data-testid="stExpander"] [class*="Header"] *,
        [data-testid="stExpander"] [data-testid*="xpander"] > div:first-child,
        [data-testid="stExpander"] [data-testid*="xpander"] > div:first-child * {{
            color:{INK} !important; -webkit-text-fill-color:{INK} !important; fill:{INK} !important; font-weight:700 !important; }}
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] summary:hover *,
        [data-testid="stExpander"] details > summary:hover * {{
            color:{TEAL_DEEP} !important; -webkit-text-fill-color:{TEAL_DEEP} !important; }}
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ background:#fff !important; }}
        [data-testid="stExpander"] p, [data-testid="stExpander"] li,
        [data-testid="stExpander"] strong {{ color:{INK} !important; -webkit-text-fill-color:{INK} !important; }}
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {{ color:{INK} !important; }}

        /* ---- tabs as bubble-pop pills (indigo active) ---- */
        .stTabs [data-baseweb="tab-list"] {{ gap:10px; border-bottom:none !important; padding:4px 0 6px; }}
        .stTabs [data-baseweb="tab"] {{
            background:#fff !important; color:{MUTED} !important; border:1px solid {BORDER} !important;
            border-radius:999px !important; padding:8px 18px !important; height:auto !important;
            box-shadow:0 4px 10px -6px rgba(40,32,90,.25), inset 0 1px 0 #fff !important;
            transition:transform .25s cubic-bezier(.2,.7,.2,1), box-shadow .25s, background .25s !important; }}
        .stTabs [data-baseweb="tab"] * {{ color:{MUTED} !important; }}
        .stTabs [data-baseweb="tab"]:hover {{ transform:translateY(-2px) !important;
            box-shadow:0 8px 16px -8px rgba(40,32,90,.3) !important; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background:#f0effb !important; border-color:#d8d4f3 !important; transform:translateY(-3px) !important;
            box-shadow:0 10px 20px -8px rgba(98,87,214,.4), inset 0 1px 0 #fff !important; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] * {{ color:{TEAL_DEEP} !important; font-weight:700 !important; }}
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none !important; background:transparent !important; }}

        [data-testid="stMetricValue"] {{ color:{INK}; }}
        [data-testid="stMetricLabel"] {{ color:{MUTED}; }}
        /* primary buttons: filled indigo */
        .stButton button[kind="primary"] {{
            background:linear-gradient(180deg,{TEAL},{TEAL_DEEP}) !important; color:#fff !important; border:none !important;
            font-weight:700 !important; border-radius:12px !important;
            box-shadow:0 8px 18px -6px rgba(98,87,214,.5), inset 0 1px 0 rgba(255,255,255,.2) !important; }}
        .stButton button[kind="primary"]:hover {{ filter:brightness(1.07); transform:translateY(-1px); }}
        /* selectbox */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background:#fff !important; border-color:{BORDER} !important; color:{INK} !important; border-radius:12px !important; }}
        [data-testid="stSelectbox"] svg {{ fill:{TEAL} !important; }}
        ul[role="listbox"] {{ background:#fff !important; border:1px solid {BORDER} !important; }}
        ul[role="listbox"] li {{ background:#fff !important; color:{INK} !important; }}
        ul[role="listbox"] li:hover {{ background:{SURFACE2} !important; color:{INK} !important; }}
        li[role="option"] {{ background:#fff !important; color:{INK} !important; }}
        li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
            background:{SURFACE2} !important; color:{TEAL_DEEP} !important; }}
        div[data-baseweb="popover"], div[data-baseweb="menu"] {{ background:#fff !important; }}
        div[data-baseweb="menu"] li {{ color:{INK} !important; }}
        /* native help tooltips — light, readable */
        [data-testid="stTooltipContent"] {{
            background:#fff !important; color:{INK} !important; border:1px solid {BORDER} !important; }}
        [data-testid="stTooltipContent"] * {{ color:{INK} !important; }}
        div[role="tooltip"] {{ background:#fff !important; color:{INK} !important; }}
        div[role="tooltip"] * {{ color:{INK} !important; }}
        [data-testid="stTooltipHoverTarget"] svg {{ fill:{MUTED} !important; opacity:0.85; }}
        [data-testid="stMetricLabel"] svg {{ fill:{TEAL} !important; opacity:0.9; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    art = (
        "<svg class='hero-art' width='160' height='116' viewBox='0 0 160 116' fill='none'>"
        "<circle cx='128' cy='42' r='74' fill='rgba(255,255,255,0.06)'/>"
        "<circle cx='150' cy='22' r='42' fill='rgba(255,255,255,0.05)'/>"
        # chat bubble (the conversation being scored)
        "<rect x='14' y='32' width='96' height='54' rx='15' fill='#fff'/>"
        "<rect x='30' y='47' width='58' height='7' rx='3.5' fill='#5046c4'/>"
        "<rect x='30' y='62' width='42' height='7' rx='3.5' fill='#b8b1f0'/>"
        "<polygon points='36,86 36,102 56,86' fill='#fff'/>"
        # score badge
        "<circle cx='118' cy='74' r='23' fill='#22b892' stroke='#fff' stroke-width='3'/>"
        "<text x='118' y='81' text-anchor='middle' font-family='Manrope,sans-serif' "
        "font-weight='800' font-size='18' fill='#fff'>87</text>"
        "</svg>"
    )
    st.markdown(
        f"""
        <div class="hero">
          {art}
          <span class="pill">LLM-as-a-judge · evaluation harness</span>
          <h1>Chat Quality Score</h1>
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
        bg, fg = "#e1f5ee", "#0f7d62"   # strong — light teal tint
    elif value >= 3:
        bg, fg = "#fbf1dc", "#9a6a14"   # ok — light amber tint
    else:
        bg, fg = "#fbe9e6", "#c1422c"   # weak — light coral tint
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
    cards = [
        ("Exact match", f"{exact:.0f}%", "Grades where judge and human gave the identical 1–5 score."),
        ("Within ±1", f"{within1:.0f}%", "Grades where they're within one point — the practical agreement bar."),
        ("CQS MAE", f"{cqs_mae:.1f} pts", "Average gap between judge and human on the 0–100 scale."),
        ("Correlation r", f"{r:.2f}", "How closely judge and human grades move together."),
    ]
    cols = st.columns(4)
    for col, (label, value, desc) in zip(cols, cards):
        col.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:12px;"
            f"padding:14px 16px;height:100%'>"
            f"<div style='font-size:0.8rem;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:0.03em'>{label}</div>"
            f"<div style='font-size:1.9rem;font-weight:800;font-family:Manrope;color:{INK};"
            f"line-height:1.1;margin:2px 0 4px'>{value}</div>"
            f"<div style='font-size:0.78rem;color:{MUTED};line-height:1.35'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


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
                    line=dict(color=SURFACE, width=lines)),
        customdata=list(zip(ids, domains)),
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                      "judge %{x:.0f} · human %{y:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", size=12, color=MUTED),
        xaxis=dict(title="LLM judge CQS →", range=[-3, 103], gridcolor=BORDER,
                   zeroline=False, dtick=25),
        yaxis=dict(title="← Human CQS", range=[-3, 103], gridcolor=BORDER,
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
        if st.button("Show full pool", type="primary"):
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
                f"<div style='background:{BORDER};border-radius:5px;height:8px;margin-top:3px'>"
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
        if st.button("Clear", type="primary"):
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
            cols[0].metric(f"{DIM_ICON[d]} {d.capitalize()}", f"{scores[d]}/5")
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
