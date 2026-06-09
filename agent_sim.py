"""A lightweight, deterministic shopping-assistant simulator.

This is NOT a language model. It reads the customer message for simple cues
(budget, size, quantity, category, intent) and assembles a plausible reply from
templates so the "Score your own" demo can generate an agent turn with zero
setup and no API key. It is intentionally simple — a stand-in for a real agent,
useful for showing the scoring mechanics end to end.
"""
from __future__ import annotations

import re

# A tiny catalog of demo products grouped by loose category keyword. Prices are
# illustrative. The simulator picks items in the customer's budget when stated.
_CATALOG = {
    "shoe": [
        ("Avia Flow running shoe", 59),
        ("Athletic Works Glide", 45),
        ("No Boundaries Lite", 72),
        ("Trailblazer Trainer", 88),
    ],
    "headphone": [
        ("SoundCore Mini buds", 29),
        ("JBL Tune wireless", 49),
        ("Anker Life Q20", 59),
    ],
    "coffee": [
        ("BrewRight drip maker", 38),
        ("AeroPress Go", 32),
        ("Hamilton Beach 12-cup", 45),
    ],
    "backpack": [
        ("DayTrek 20L pack", 34),
        ("Urban Commuter bag", 52),
        ("TrailLite hiking pack", 78),
    ],
    "blender": [
        ("NutriMix personal blender", 36),
        ("Ninja Fit single-serve", 59),
        ("Oster 6-cup", 44),
    ],
}

# Keyword -> category, so we can map varied phrasings to a catalog bucket.
_CATEGORY_HINTS = {
    "shoe": "shoe", "sneaker": "shoe", "running": "shoe", "trainer": "shoe",
    "headphone": "headphone", "earbud": "headphone", "earphone": "headphone", "buds": "headphone",
    "coffee": "coffee", "espresso": "coffee", "brew": "coffee",
    "backpack": "backpack", "bag": "backpack", "pack": "backpack",
    "blender": "blender", "smoothie": "blender",
}


def _extract_budget(text: str) -> int | None:
    # "under $80", "below 50", "less than $30", "$25 budget"
    m = re.search(r"(?:under|below|less than|max|up to)\s*\$?\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\$\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _extract_size(text: str) -> str | None:
    m = re.search(r"size\s*(\d+(?:\.\d+)?)", text, re.I)
    return m.group(1) if m else None


def _detect_category(text: str) -> str | None:
    low = text.lower()
    for hint, cat in _CATEGORY_HINTS.items():
        if hint in low:
            return cat
    return None


def simulate_agent_reply(customer_message: str) -> str:
    """Return a plausible shopping-assistant reply to the customer message.

    Deterministic and rule-based — same input always yields the same output.
    """
    msg = (customer_message or "").strip()
    if not msg:
        return (
            "Hi! I'm here to help you find what you need. Could you tell me a bit "
            "more about what you're shopping for, and any budget or preferences?"
        )

    budget = _extract_budget(msg)
    size = _extract_size(msg)
    category = _detect_category(msg)

    # In-domain: we have a matching product category.
    if category and category in _CATALOG:
        items = _CATALOG[category]
        if budget is not None:
            items = [it for it in items if it[1] <= budget] or items[:2]
        picks = items[:3]
        listing = ", ".join(f"the {name} (${price})" for name, price in picks)

        size_clause = f" in size {size}" if size else ""
        budget_clause = f" under ${budget}" if budget is not None else ""
        return (
            f"Here are a few {category} options{budget_clause}{size_clause}: "
            f"{listing}. Want me to add one of these to your cart, or see more choices?"
        )

    # Out-of-domain (e.g. recipes, services): be honest and redirect, which is
    # realistic agent behavior and gives the judge something fair to score.
    budget_clause = f" within your ${budget} budget" if budget is not None else ""
    return (
        f"I can help you shop for products{budget_clause}, but I'm focused on store "
        "items rather than that kind of request. If you can tell me a product "
        "category you're interested in, I'll pull up some specific options for you."
    )
