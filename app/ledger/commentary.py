"""Short written commentary on the daily picks from Claude.

Uses the Anthropic API when ANTHROPIC_API_KEY is set, otherwise OpenRouter when OPENROUTER_API_KEY is set
(model from OPENROUTER_MODEL, default Claude Opus 5.5). Without a key the scan still runs and simply has
no commentary.
"""
from __future__ import annotations

import json
import os

MODEL = "claude-opus-5-5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-opus-5.5"
NO_KEY = ("No API key found: set OPENROUTER_API_KEY (OpenRouter) or ANTHROPIC_API_KEY (Anthropic) to get "
          "Claude's commentary.")

SYSTEM = """You write a short morning note for one private investor who uses a personal portfolio tracker.
The tracker has already screened index members (S&P 500, Nasdaq 100, Euro Stoxx 50, FTSE MIB) on price trend
and momentum, and ranked a second list by how well each stock would diversify the investor's current portfolio.
You receive those results as JSON.

Write in English, plain Markdown, at most about 800 words, with these sections:
## Market read - two or three sentences on what the lists say together (sectors and regions leading, how broad the trend is).
## Momentum picks - one bullet per stock: what the company does, why its numbers stand out, the main risk to watch.
## Portfolio-fit picks - one bullet per stock: what it would add to this portfolio and the main risk.
## Watch-outs - two or three bullets (concentration among picks, stretched RSI, currency and fee effects).
If long_shot_picks are present, add before Watch-outs:
## Long shots - stocks far below the analysts' average target, meant only for small, high-risk stakes. One bullet
per stock: what the company does, why analysts may be right, why the market may disagree (flag falling_knife
stocks), the main risk. Say plainly that analyst targets are often too optimistic.

Rules: rely on the numbers given and on general knowledge of the companies. You have no live news and your
knowledge has a cutoff, so do not state recent events, earnings or prices as fact. Do not predict prices.
This is a screen to study, not personal advice: phrase points as things to check, not instructions to buy."""


def provider() -> str:
    """"anthropic", "openrouter" or "" when no key is set. A direct Anthropic key wins when both exist."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return ""


def available() -> bool:
    return bool(provider())


def _openrouter(user_message: str) -> tuple[str, str]:
    import requests

    model = os.environ.get("OPENROUTER_MODEL") or OPENROUTER_MODEL
    try:
        r = requests.post(
            OPENROUTER_URL, timeout=180,
            headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"].strip(), "X-Title": "Stock Ledger"},
            json={"model": model, "max_tokens": 8000,
                  "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}]})
    except requests.RequestException:
        return "", "Could not reach OpenRouter (network)."
    try:
        body = r.json()
    except ValueError:
        body = {}
    error = body.get("error") or {}
    if r.status_code == 401:
        return "", "The OpenRouter API key was rejected. Check OPENROUTER_API_KEY."
    if r.status_code == 402:
        return "", "OpenRouter: not enough credits for this request. Add credits on openrouter.ai."
    if r.status_code == 429:
        return "", "OpenRouter rate limit reached; commentary skipped today."
    if r.status_code != 200 or error:
        return "", f"OpenRouter error {error.get('code', r.status_code)}: {error.get('message', r.reason)}"
    choice = (body.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    if not text:
        return "", "OpenRouter returned no text for today's picks."
    if choice.get("finish_reason") == "length":
        text += "\n\n*(Commentary was cut off.)*"
    u = body.get("usage") or {}
    return text, (f"Written by {body.get('model', model)} via OpenRouter ({u.get('prompt_tokens', '?')} input / "
                  f"{u.get('completion_tokens', '?')} output tokens).")


def _slim(picks: list[dict], keys: list[str]) -> list[dict]:
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k in keys} for p in picks]


def write_commentary(standalone: list[dict], fit: list[dict], portfolio: dict,
                     long_shots: list[dict] | None = None) -> tuple[str, str]:
    """Return (markdown, note). markdown is "" when no commentary could be produced; note says why."""
    if not available():
        return "", NO_KEY

    keys = ["ticker", "name", "sector", "country", "currency", "price", "r1m", "r3m", "r6m", "r12_1", "rsi",
            "vol", "below_high", "score", "held"]
    payload = {
        "portfolio": portfolio,
        "momentum_picks": _slim(standalone, keys),
        "portfolio_fit_picks": _slim(fit, keys + ["fit", "corr", "sector_weight", "country_weight", "eff_n_after"]),
        **({"long_shot_picks": _slim(long_shots, ["ticker", "name", "sector", "country", "currency", "price",
                                                  "target_mean", "target_low", "target_high", "upside", "n_analysts",
                                                  "rating", "vol", "r3m", "below_high", "falling_knife", "held"])}
           if long_shots else {}),
        "field_notes": {"r12_1": "12-month return excluding the last month", "vol": "annualised volatility",
                        "below_high": "distance from 1-year high", "score": "momentum percentile 0-100",
                        "fit": "portfolio fit 0-100", "corr": "correlation of daily returns with the portfolio",
                        "upside": "average analyst target vs price", "target_low/high": "lowest/highest analyst target"},
    }
    user_message = "Today's screen results:\n" + json.dumps(payload, default=str)
    if provider() == "openrouter":
        return _openrouter(user_message)
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            system=SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        return "", "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return "", "Anthropic API rate limit reached; commentary skipped today."
    except anthropic.APIStatusError as e:
        return "", f"Anthropic API error {e.status_code}: {e.message}"
    except anthropic.APIConnectionError:
        return "", "Could not reach the Anthropic API (network)."
    if response.stop_reason == "refusal":
        return "", "Claude declined to write commentary for today's picks."
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n*(Commentary was cut off.)*"
    u = response.usage
    return text, f"Written by {response.model} ({u.input_tokens} input / {u.output_tokens} output tokens)."
