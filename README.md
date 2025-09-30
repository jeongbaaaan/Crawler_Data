# Gym Price Crawler (KR)

Seed-driven crawler that finds gym price pages (BFS + heuristics), extracts Korean price phrases, and normalizes to KRW.
- Robust fetch with charset sniffing (fixes Korean garbling)
- Parsers: domain-specific → general regex → BFS fallback
- Outputs: `data/processed/gym_prices.csv`, debug HTML in `data/debug_html/`
