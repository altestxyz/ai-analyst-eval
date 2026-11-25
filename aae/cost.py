import pandas as pd

from aae.models import load_models

MODELS = load_models()
TOKENS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")


def _price(model, key, fallback=None):
    c = MODELS.get(model, {})
    v = c.get(key, c.get(fallback, 0) if fallback else 0)
    return float(v or 0)


def add_costs(df):
    if df.empty or "model" not in df:
        return df
    df = df.copy()
    for t in TOKENS:
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0) if t in df else 0.0
    p_in = df.model.map(lambda m: _price(m, "price_in"))
    p_out = df.model.map(lambda m: _price(m, "price_out"))
    p_cr = df.model.map(lambda m: _price(m, "price_cache_read", "price_in"))
    p_cw = df.model.map(lambda m: _price(m, "price_cache_write", "price_in"))
    df["cost_usd"] = (df.input_tokens * p_in + df.cache_read_tokens * p_cr + df.cache_write_tokens * p_cw + df.output_tokens * p_out) / 1e6
    df["cost_list_usd"] = ((df.input_tokens + df.cache_read_tokens + df.cache_write_tokens) * p_in + df.output_tokens * p_out) / 1e6
    return df


def price_note(df):
    lines = ["Prices used, US$ per million tokens (edit FRONTIER in the config to change them; every run is re-costed):"]
    for m in sorted(df.model.unique()):
        c = MODELS.get(m, {})
        if c.get("provider") == "ollama":
            continue
        pin, pout = _price(m, "price_in"), _price(m, "price_out")
        lines.append(
            f"  {m} {c.get('label', '')}: {pin} in, {pout} out, {_price(m, 'price_cache_read', 'price_in')} cached"
            + ("   <- NO PRICE SET, its cost reads 0" if not (pin and pout) else "")
        )
        no_tok = int(((df.model == m) & (df.input_tokens == 0) & (df.output_tokens == 0)).sum())
        if no_tok:
            lines.append(f"     {no_tok} of its runs logged no token counts (sdk proxy mode), so their cost reads 0")
    return "\n".join(lines)
