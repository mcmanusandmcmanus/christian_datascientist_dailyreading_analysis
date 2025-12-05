# Catholic Daily Reading Graph MVP

Mobile-friendly MVP to scrape USCCB daily readings, fetch Catholic translation text, build NetworkX context graphs, and surface Catholic-flavored NLP signals (rosary resonance, hotspot words). Includes a lightweight HTML view tuned for phone screens.

## Stack
- Python pipeline: `requests`, `beautifulsoup4`, `networkx`, `matplotlib`.
- Data outputs: JSON payload, GEXF graph, PNG visuals.
- Frontend: static HTML (`frontend/mobile_mvp.html`) that reads the generated JSON/PNGs.

## Setup (API mode)
1) Python deps: `pip install -r requirements.txt`
2) Env vars (scripture.api.bible):
   - `SCRIPTURE_API_KEY`
   - `BIBLE_ID` (Catholic-friendly, e.g., NABRE/GNTCE from scripture.api.bible)
3) Run pipeline (today):
   - `python scripts/usccb_daily_pipeline.py`
   - Backfill: `python scripts/usccb_daily_pipeline.py --date YYYY-MM-DD`

## Setup (offline, Douay-Rheims 1899)
1) Download DR1899 CSV (e.g., from the scrollmapper/bible_databases repo) with columns `book,chapter,verse,text` to `data/drb_1899.csv`.
2) Run pipeline offline:
   - `python scripts/usccb_daily_pipeline.py --mode offline --offline-path data/drb_1899.csv`

## Outputs
- `data/latest_payload.json` — readings, text, top words, rosary resonance, sources.
- `output/hotwords.png` — top word frequencies.
- `output/context_graph.png` — reading <-> word graph (edge weight >= 2).
- `output/context_graph.gexf` — graph for Gephi/Cytoscape exploration.

## View on mobile/desktop
- Serve the repo root: `python -m http.server 8000`
- Open `http://localhost:8000/frontend/mobile_mvp.html`

## Automation (daily refresh)
- Windows Task Scheduler (6am example):
  - Action: `python C:\Users\mcman\webapp_mobile_christiands\scripts\usccb_daily_pipeline.py`
  - Start in: `C:\Users\mcman\webapp_mobile_christiands`
  - Ensure env vars are available to the task user.
- Linux/macOS cron:
  - API: `0 6 * * * cd /path/webapp_mobile_christiands && SCRIPTURE_API_KEY=... BIBLE_ID=... /usr/bin/python scripts/usccb_daily_pipeline.py`
  - Offline: `0 6 * * * cd /path/webapp_mobile_christiands && /usr/bin/python scripts/usccb_daily_pipeline.py --mode offline --offline-path data/drb_1899.csv`

## Catholic flavor
- Rosary resonance: counts mystery keywords (Joyful/Luminous/Sorrowful/Glorious) across the day’s readings.
- Color palette and copy tuned toward Catholic prayer posture.
- Sources & citations block keeps USCCB and translation info visible.

## Extending
- Add spaCy lemmatization in `tokenize` for better NLP.
- Add sentiment/embedding clustering for "old in new / new in old" echoes.
- Deploy to Render: serve `frontend` static; run pipeline on boot/cron to refresh `data/output`.
- iOS/React Native: consume `data/latest_payload.json`; render PNGs or derive graphs client-side (d3-force).

## One cool feature idea
- "Rosary heatline": animate bead-by-bead counts from the resonance data to show which mysteries light up in today's readings (future front-end enhancement).
