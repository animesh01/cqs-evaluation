"""A lightweight, deterministic shopping-assistant simulator.

This is NOT a language model. It reads the customer message for simple cues
(budget, size, quantity, category, intent) and assembles a plausible reply from
templates so the "Score your own" demo can generate an agent turn with zero
setup and no API key.

Coverage is intentionally broad — ~25 product categories — and when the message
doesn't match a known category, a noun-extraction fallback still produces a
tailored reply about the requested item, rather than a flat redirect. A true
redirect is reserved only for clearly non-shopping requests (weather, recipes,
directions, etc.). Product names are generic placeholders (Brand A / B / C
style) and all prices are illustrative.
"""
from __future__ import annotations

import re

# Generic placeholder products per category: (name, price). Three tiers each.
_CATALOG = {
    "running shoe": [("Brand A Lightweight Runner", 59), ("Brand B Daily Trainer", 72), ("Brand C Trail Runner", 88)],
    "headphone": [("Brand A Mini Buds", 29), ("Brand B Wireless", 49), ("Brand C Pro ANC", 89)],
    "coffee maker": [("Brand A Drip Maker", 38), ("Brand B Pour-Over Kit", 32), ("Brand C 12-Cup", 59)],
    "backpack": [("Brand A 20L Daypack", 34), ("Brand B Commuter", 52), ("Brand C Hiking Pack", 78)],
    "blender": [("Brand A Personal Blender", 36), ("Brand B Single-Serve", 44), ("Brand C High-Power", 69)],
    "laptop": [("Brand A Air 14", 649), ("Brand B Studio 15", 899), ("Brand C Budget 14", 449)],
    "monitor": [("Brand A 24\" FHD", 129), ("Brand B 27\" QHD", 219), ("Brand C 32\" 4K", 349)],
    "keyboard": [("Brand A Quiet 75%", 89), ("Brand B TKL Mechanical", 72), ("Brand C Full-Size", 59)],
    "standing desk": [("Brand A 48\" Electric", 249), ("Brand B 55\" Pro", 289), ("Brand C Compact", 199)],
    "office chair": [("Brand A Ergo Mesh", 179), ("Brand B Task Chair", 129), ("Brand C Executive", 259)],
    "yoga mat": [("Brand A Natural Rubber", 45), ("Brand B Cork-Top", 52), ("Brand C Travel Mat", 29)],
    "water bottle": [("Brand A Insulated 24oz", 24), ("Brand B Glass 18oz", 19), ("Brand C Sport 32oz", 16)],
    "robot vacuum": [("Brand A Pet R3", 219), ("Brand B Go", 179), ("Brand C Mapping Pro", 299)],
    "tv": [("Brand A 50\" 4K", 329), ("Brand B 55\" QLED", 499), ("Brand C 43\" FHD", 249)],
    "phone case": [("Brand A Slim Shell", 18), ("Brand B Rugged Armor", 29), ("Brand C Clear", 14)],
    "charger": [("Brand A 20W USB-C", 19), ("Brand B 3-Port GaN", 39), ("Brand C Wireless Pad", 29)],
    "smartwatch": [("Brand A Fit Lite", 99), ("Brand B Active 2", 169), ("Brand C Pro GPS", 249)],
    "cookware set": [("Brand A 8-Piece Nonstick", 89), ("Brand B Stainless 10-Pc", 149), ("Brand C Cast Iron", 69)],
    "protein powder": [("Brand A Whey Vanilla", 32), ("Brand B Plant Chocolate", 34), ("Brand C Isolate", 45)],
    "skincare": [("Brand A Gentle Cleanser", 16), ("Brand B Vitamin C Serum", 28), ("Brand C Moisturizer", 22)],
    "toy": [("Brand A STEM Kit", 24), ("Brand B Building Blocks", 29), ("Brand C Craft Set", 19)],
    "desk lamp": [("Brand A LED Task Lamp", 34), ("Brand B Clamp Lamp", 27), ("Brand C Ring Light", 45)],
    "tablet": [("Brand A 10\" Lite", 179), ("Brand B 11\" Pro", 399), ("Brand C 8\" Mini", 149)],
    "speaker": [("Brand A Portable", 39), ("Brand B Bookshelf Pair", 129), ("Brand C Smart Speaker", 49)],
    "jacket": [("Brand A Rain Shell", 59), ("Brand B Insulated Parka", 119), ("Brand C Fleece", 39)],
}

# Map varied phrasings/keywords to a canonical catalog category.
_CATEGORY_HINTS = {
    "running shoe": ["running shoe", "running shoes", "sneaker", "trainer", "runners", "shoe"],
    "headphone": ["headphone", "headphones", "earbud", "earbuds", "earphone", "buds"],
    "coffee maker": ["coffee maker", "coffee machine", "espresso", "coffee", "french press", "pour over"],
    "backpack": ["backpack", "rucksack", "daypack", "book bag"],
    "blender": ["blender", "smoothie maker"],
    "laptop": ["laptop", "notebook computer", "ultrabook", "macbook", "chromebook"],
    "monitor": ["monitor", "display", "screen"],
    "keyboard": ["keyboard", "mechanical keyboard"],
    "standing desk": ["standing desk", "sit-stand desk", "adjustable desk", "desk"],
    "office chair": ["office chair", "desk chair", "ergonomic chair", "task chair"],
    "yoga mat": ["yoga mat", "exercise mat", "workout mat"],
    "water bottle": ["water bottle", "tumbler", "flask", "hydration"],
    "robot vacuum": ["robot vacuum", "robovac", "vacuum", "roomba"],
    "tv": ["tv", "television", "4k tv", "smart tv"],
    "phone case": ["phone case", "iphone case", "case for", "phone cover"],
    "charger": ["charger", "charging", "power adapter", "power bank"],
    "smartwatch": ["smartwatch", "smart watch", "fitness tracker", "fitness band"],
    "cookware set": ["cookware", "pots and pans", "frying pan", "saucepan", "skillet"],
    "protein powder": ["protein powder", "protein shake", "whey", "protein"],
    "skincare": ["skincare", "moisturizer", "serum", "cleanser", "sunscreen", "face cream"],
    "toy": ["toy", "toys", "kids gift", "children's gift", "lego", "gift for a", "gift for", "birthday gift"],
    "desk lamp": ["desk lamp", "lamp", "reading light", "task light"],
    "tablet": ["tablet", "ipad"],
    "speaker": ["speaker", "speakers", "bluetooth speaker", "soundbar"],
    "jacket": ["jacket", "coat", "parka", "windbreaker", "raincoat"],
}

# Clearly non-shopping intents → honest redirect (the only time we redirect).
_OUT_OF_SCOPE = [
    "weather", "recipe", "directions", "translate", "joke", "news", "stock price",
    "score", "definition", "meaning of", "how do i cook", "how to cook",
]

# Words to ignore when guessing the shopping noun in the fallback.
_STOP = {
    "i", "need", "want", "looking", "for", "a", "an", "the", "some", "to", "buy",
    "get", "find", "me", "my", "please", "good", "best", "great", "with", "and",
    "under", "below", "less", "than", "around", "about", "that", "can", "you",
    "help", "is", "are", "of", "in", "on", "it", "would", "like", "recommend",
    "recommendation", "recommendations", "suggest", "show", "any", "have", "do",
    "cheap", "affordable", "new", "quality", "size", "budget", "price", "cost",
}
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def _extract_budget(text: str):
    m = re.search(r"(?:under|below|less than|max|up to|around|about)\s*\$?\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\$\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _extract_size(text: str):
    m = re.search(r"size\s*(\d+(?:\.\d+)?)", text, re.I)
    return m.group(1) if m else None


def _detect_category(text: str):
    low = text.lower()
    # Prefer the longest matching hint phrase so "running shoe" beats "shoe".
    best = None
    best_len = 0
    for cat, hints in _CATEGORY_HINTS.items():
        for h in hints:
            if h in low and len(h) > best_len:
                best, best_len = cat, len(h)
    return best


def _is_out_of_scope(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _OUT_OF_SCOPE)


def _guess_noun(text: str):
    """Pick the most likely shopping noun from the message for the fallback.

    Heuristic: the product usually follows an intent phrase ("looking for a X",
    "need a X", "want a X", "recommend a X"). We grab the first meaningful word
    after such a trigger; if none is found, fall back to the first meaningful
    word overall. We also stop at prepositions like "for"/"with" so qualifiers
    ("a hammock for my balcony") don't override the real product ("hammock").
    """
    low = text.lower()
    triggers = [
        "looking for a", "looking for an", "looking for", "i need a", "i need an",
        "i need", "i want a", "i want an", "i want", "need a", "need an", "want a",
        "want an", "recommend a", "recommend an", "recommend", "suggest a",
        "do you have", "do you sell", "shopping for a", "shopping for",
    ]
    segment = low
    for trig in triggers:
        idx = low.find(trig)
        if idx != -1:
            segment = low[idx + len(trig):]
            break

    boundary = re.compile(r"\b(for|with|that|under|below|around|to|in|on)\b")
    bm = boundary.search(segment)
    if bm:
        segment = segment[: bm.start()]

    words = [w for w in _WORD_RE.findall(segment) if w not in _STOP and len(w) > 2]
    if words:
        return words[-1]

    # Fallback: first meaningful word anywhere in the message.
    allw = [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP and len(w) > 2]
    return allw[0] if allw else None


def simulate_agent_reply(customer_message: str) -> str:
    msg = (customer_message or "").strip()
    if not msg:
        return (
            "Hi! I'm here to help you find what you need. Could you tell me a bit "
            "more about what you're shopping for, and any budget or preferences?"
        )

    budget = _extract_budget(msg)
    size = _extract_size(msg)

    # 1) Known category → specific product picks.
    category = _detect_category(msg)
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

    # 2) Genuinely non-shopping request → honest redirect.
    if _is_out_of_scope(msg):
        return (
            "That's outside what I can help with — I'm a shopping assistant for store "
            "products. If there's an item you're shopping for, tell me the category and "
            "I'll pull up some options."
        )

    # 3) Unknown product → noun-extraction fallback (tailored, not a flat redirect).
    noun = _guess_noun(msg)
    if noun:
        budget_clause = f" under ${budget}" if budget is not None else ""
        size_clause = f" in size {size}" if size else ""
        return (
            f"Here are a few {noun} options{budget_clause}{size_clause} I'd suggest: "
            f"the Brand A {noun.capitalize()} (good value), the Brand B {noun.capitalize()} "
            f"(mid-range), and the Brand C {noun.capitalize()} (premium). Want me to compare "
            f"them or add one to your cart?"
        )

    # 4) Last resort.
    return (
        "I can help you shop for that. Could you tell me a bit more — a product type and "
        "any budget — so I can pull up specific options?"
    )
