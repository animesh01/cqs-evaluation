# Chat Quality Score (CQS) — Streamlit app

An **LLM-as-a-judge** evaluation demo for conversational AI. It scores customer
conversations on four rubric dimensions — relevance, helpfulness, correctness,
tone — rolls them into a single **0–100 CQS**, and calibrates the automated
judge against human labels.

## Data Disclaimer

All conversations, scores, and examples in this repository are **synthetic and fabricated for demonstration**. Nothing here is derived from any employer, customer, or production system, and the repository contains no proprietary, confidential, or company-specific data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens in **Mock** mode (a deterministic keyword heuristic) and runs with
no API key. To use the real Anthropic judge, pick **Real LLM judge** in the
sidebar and paste an Anthropic API key — it lives only in your session.

## Tabs

- **Benchmark** — scores a fixed, human-labeled set and reports judge↔human
  agreement (exact match, within ±1, CQS MAE, correlation) with a per-dimension
  breakdown.
- **Score your own** — paste a customer/assistant exchange and grade it live.
- **Rubric** — the four dimensions and the two-tier (human + LLM) design.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, **New app**, select the repo/branch and set
   the main file to `streamlit_app.py`.
3. (Optional) In the app's **Settings → Secrets**, add your key so you don't have
   to paste it each time:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Never commit a key to the repo — `.streamlit/secrets.toml` is git-ignored.

## Files

```
streamlit_app.py    # the app
cqs_judge.py        # rubric, mock judge, and real Anthropic judge
evaluate.py         # original command-line evaluator (still works)
data/conversation_pool.json
```
