"""
USCCB daily readings scraper -> NetworkX graph + Catholic-themed NLP signals.

Modes:
- api (default): scripture.api.bible (NABRE/GNTCE/etc).
- offline: Douay-Rheims 1899 CSV (local) without network.

Features:
- Scrapes daily reading citations from bible.usccb.org.
- Fetches passage text from scripture.api.bible or a local DR1899 CSV.
- Builds word/reading graphs (NetworkX) and hotspot visualizations (matplotlib).
- Adds rosary resonance stats (mystery keyword counts) for a Catholic focus.
- Saves JSON payload, GEXF graph, and PNGs for mobile/web consumption.

Env vars (API mode):
    SCRIPTURE_API_KEY  (required)
    BIBLE_ID           (preferred NABRE/GNTCE id from scripture.api.bible)

CLI examples:
    python scripts/usccb_daily_pipeline.py --date 2024-07-16
    python scripts/usccb_daily_pipeline.py --mode offline --offline-path data/drb_1899.csv
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import requests
from bs4 import BeautifulSoup


STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "a",
    "that",
    "for",
    "is",
    "on",
    "with",
    "as",
    "be",
    "are",
    "by",
    "he",
    "she",
    "they",
    "you",
    "i",
    "we",
    "it",
    "his",
    "her",
    "their",
    "our",
    "from",
    "this",
    "these",
    "those",
    "an",
    "at",
    "or",
    "but",
    "if",
    "so",
    "not",
    "was",
}

ROSARY_MYSTERIES = {
    "joyful": {"annunciation", "visitation", "nativity", "presentation", "finding"},
    "luminous": {"baptism", "cana", "proclamation", "transfiguration", "eucharist"},
    "sorrowful": {"gethsemane", "scourging", "thorns", "calvary", "crucifixion"},
    "glorious": {"resurrection", "ascension", "pentecost", "assumption", "coronation"},
}

# Common USCCB abbreviations to canonical names (Douay-Rheims 1899 style)
ABBR_MAP = {
    "gn": "Genesis",
    "ex": "Exodus",
    "lv": "Leviticus",
    "nm": "Numbers",
    "dt": "Deuteronomy",
    "jos": "Joshua",
    "jgs": "Judges",
    "ru": "Ruth",
    "1 sm": "1 Samuel",
    "2 sm": "2 Samuel",
    "1 kgs": "1 Kings",
    "2 kgs": "2 Kings",
    "1 chr": "1 Chronicles",
    "2 chr": "2 Chronicles",
    "neh": "Nehemiah",
    "tb": "Tobit",
    "jdt": "Judith",
    "est": "Esther",
    "1 mc": "1 Maccabees",
    "2 mc": "2 Maccabees",
    "jb": "Job",
    "ps": "Psalms",
    "prv": "Proverbs",
    "eccl": "Ecclesiastes",
    "sir": "Sirach",
    "wis": "Wisdom",
    "is": "Isaiah",
    "jer": "Jeremiah",
    "lam": "Lamentations",
    "bar": "Baruch",
    "ez": "Ezekiel",
    "dn": "Daniel",
    "hos": "Hosea",
    "jl": "Joel",
    "am": "Amos",
    "ob": "Obadiah",
    "jon": "Jonah",
    "mi": "Micah",
    "na": "Nahum",
    "hab": "Habakkuk",
    "zep": "Zephaniah",
    "hag": "Haggai",
    "zec": "Zechariah",
    "mal": "Malachi",
    "mt": "Matthew",
    "mk": "Mark",
    "lk": "Luke",
    "jn": "John",
    "acts": "Acts",
    "rom": "Romans",
    "1 cor": "1 Corinthians",
    "2 cor": "2 Corinthians",
    "gal": "Galatians",
    "eph": "Ephesians",
    "phil": "Philippians",
    "col": "Colossians",
    "1 thes": "1 Thessalonians",
    "2 thes": "2 Thessalonians",
    "1 tm": "1 Timothy",
    "2 tm": "2 Timothy",
    "ti": "Titus",
    "phlm": "Philemon",
    "heb": "Hebrews",
    "jas": "James",
    "1 pt": "1 Peter",
    "2 pt": "2 Peter",
    "1 jn": "1 John",
    "2 jn": "2 John",
    "3 jn": "3 John",
    "jude": "Jude",
    "rv": "Revelation",
}


@dataclass
class Passage:
    reference: str
    text: str


def request_with_retry(url: str, *, headers=None, params=None, timeout: int = 15, retries: int = 3, backoff: float = 1.5):
    headers = headers or {}
    params = params or {}
    last_err = None
    for attempt in range(retries):
        try:
            res = requests.get(url, headers=headers, params=params, timeout=timeout)
            res.raise_for_status()
            return res
        except Exception as err:
            last_err = err
            sleep_for = backoff ** attempt
            time.sleep(sleep_for)
    raise last_err


def fetch_usccb_refs(target_iso: str) -> List[str]:
    url = f"https://bible.usccb.org/daily-bible-readings?date={target_iso}"
    res = request_with_retry(url, timeout=20)
    soup = BeautifulSoup(res.text, "html.parser")
    body = " ".join(el.get_text(" ", strip=True) for el in soup.select("article, section"))
    refs = re.findall(r"([1-3]?\s?[A-Za-z]+\s+\d+:\d+(?:-\d+)?(?:,\s*\d+:\d+)*)", body)

    if not refs:
        links = soup.select("a")
        for a in links:
            href = a.get("href", "")
            text = a.get_text(" ", strip=True)
            if "/bible/" in href and ":" in text:
                refs.append(text)

    refs = [" ".join(r.split()) for r in refs]
    return sorted(set(refs))


def fetch_passage_from_api(reference: str, bible_id: str, api_key: str) -> Passage:
    headers = {"api-key": api_key}
    params = {"reference": reference, "content-type": "text"}
    url = f"https://api.scripture.api.bible/v1/bibles/{bible_id}/passages"
    res = request_with_retry(url, headers=headers, params=params, timeout=20)
    data = res.json().get("data", {})
    text = BeautifulSoup(data.get("content", ""), "html.parser").get_text(" ", strip=True)
    return Passage(reference=reference, text=text)


def tokenize(txt: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z']+", txt.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def build_graph(passages: Dict[str, Passage]) -> Tuple[nx.Graph, collections.Counter]:
    G = nx.Graph()
    totals: collections.Counter = collections.Counter()
    max_pairs = 500
    for ref, passage in passages.items():
        words = tokenize(passage.text)
        totals.update(words)
        G.add_node(ref, kind="reading")
        counts = collections.Counter(words)
        for w, c in counts.items():
            G.add_node(w, kind="word")
            G.add_edge(ref, w, weight=c)
        if len(words) <= max_pairs:
            for w1, w2 in itertools.combinations(words, 2):
                if w1 == w2:
                    continue
                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += 1
                else:
                    G.add_edge(w1, w2, weight=1)
    return G, totals


def rosary_resonance(tokens: List[str]) -> Dict[str, int]:
    counts = {}
    token_set = collections.Counter(tokens)
    for mystery, keywords in ROSARY_MYSTERIES.items():
        counts[mystery] = sum(token_set[k] for k in keywords)
    counts["total_mentions"] = sum(counts.values())
    return counts


def plot_hotwords(totals: collections.Counter, out_path: Path, top: int = 15) -> None:
    common = totals.most_common(top)
    if not common:
        return
    words, counts = zip(*common)
    plt.figure(figsize=(10, 4))
    plt.bar(words, counts, color="#d4af37")  # gold, Catholic vibe
    plt.xticks(rotation=45, ha="right")
    plt.title("Hotspot words")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_graph(G: nx.Graph, out_path: Path, min_weight: int = 2) -> None:
    H = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0) >= min_weight)
    if not H.nodes:
        return
    pos = nx.spring_layout(H, k=0.5, weight="weight", seed=7)
    colors = ["#8b1e3f" if H.nodes[n].get("kind") == "reading" else "#1c4c96" for n in H.nodes]
    nx.draw_networkx(
        H,
        pos,
        node_color=colors,
        with_labels=True,
        font_size=8,
        node_size=650,
        edge_color="#9bb4d1",
    )
    plt.title("Reading-word context graph")
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_gexf(G: nx.Graph, out_path: Path) -> None:
    nx.write_gexf(G, out_path)


def canon_book(name: str) -> str:
    key = name.strip().lower().replace(".", "")
    if key in ABBR_MAP:
        return ABBR_MAP[key]
    return name.title()


def parse_reference(reference: str) -> Tuple[str, List[Tuple[int, List[int]]]]:
    """
    Returns canonical book name and list of (chapter, [verses]).
    Supports forms like "John 3:16-18, 20" and "1 Kgs 3:4-13".
    Cross-chapter spans are split into start and end chapters (best-effort).
    """
    m = re.match(r"([1-3]?\s?[A-Za-z. ]+)\s+(.+)", reference)
    if not m:
        raise ValueError(f"Cannot parse reference: {reference}")
    book_raw, rest = m.group(1), m.group(2)
    book = canon_book(book_raw)
    segments = re.split(r";\s*", rest)
    chapter_verses = []
    for seg in segments:
        if ":" not in seg:
            continue
        chap_str, verses_str = seg.split(":", 1)
        chap = int(chap_str.strip())
        verses = []
        for part in verses_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                if ":" in end:
                    end_chap_str, end_verse_str = end.split(":")
                    end_chap = int(end_chap_str)
                    end_verse = int(end_verse_str)
                    start_verse = int(start)
                    verses.extend(list(range(start_verse, start_verse + 200)))
                    chapter_verses.append((end_chap, list(range(1, end_verse + 1))))
                    continue
                verses.extend(list(range(int(start), int(end) + 1)))
            else:
                try:
                    verses.append(int(part))
                except ValueError:
                    continue
        if verses:
            chapter_verses.append((chap, verses))
    return book, chapter_verses


def load_dr1899_csv(path: Path) -> Dict[Tuple[str, int, int], str]:
    """
    Expect CSV with columns: book,chapter,verse,text
    Returns dict keyed by (book_lower, chapter, verse).
    """
    import csv

    if not path.exists():
        raise FileNotFoundError(f"Offline Bible not found: {path}")
    required = {"book", "chapter", "verse", "text"}
    index: Dict[Tuple[str, int, int], str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = required.difference({c.lower() for c in reader.fieldnames or []})
        if missing_cols:
            raise ValueError(f"Offline CSV missing columns: {', '.join(sorted(missing_cols))}")
        for row in reader:
            book = row.get("book") or row.get("book_name") or ""
            chapter = row.get("chapter") or row.get("chapter_number") or ""
            verse = row.get("verse") or row.get("verse_number") or ""
            text = row.get("text") or row.get("verse_text") or ""
            try:
                key = (book.strip().lower(), int(chapter), int(verse))
            except ValueError:
                continue
            index[key] = text.strip()
    return index


def fetch_passage_offline(reference: str, verse_index: Dict[Tuple[str, int, int], str]) -> Tuple[Passage, int]:
    book, chap_map = parse_reference(reference)
    book_key = book.strip().lower()
    verses_out = []
    missing = 0
    for chap, verses in chap_map:
        for v in verses:
            text = verse_index.get((book_key, chap, v))
            if text:
                verses_out.append(f"{chap}:{v} {text}")
            else:
                missing += 1
    return Passage(reference=reference, text=" ".join(verses_out)), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="USCCB daily readings -> graph + viz")
    parser.add_argument("--date", dest="date_str", help="ISO date YYYY-MM-DD (default: today)")
    parser.add_argument("--bible-id", dest="bible_id", help="scripture.api.bible BIBLE_ID")
    parser.add_argument(
        "--mode",
        choices=["api", "offline"],
        default="api",
        help="api (scripture.api.bible) or offline (DR 1899 CSV)",
    )
    parser.add_argument(
        "--offline-path",
        default="data/drb_1899.csv",
        help="path to Douay-Rheims 1899 CSV (offline mode)",
    )
    parser.add_argument(
        "--output-dir", default="output", help="directory to write png/gexf outputs (default: output)"
    )
    parser.add_argument(
        "--data-dir", default="data", help="directory to write JSON payloads (default: data)"
    )
    args = parser.parse_args()

    target_date = args.date_str or date.today().isoformat()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("SCRIPTURE_API_KEY")
    bible_id = args.bible_id or os.environ.get("BIBLE_ID")

    refs = fetch_usccb_refs(target_date)
    passages: Dict[str, Passage] = {}
    offline_missing: Dict[str, int] = {}
    total_missing = 0
    total_requested = 0

    if args.mode == "offline":
        verse_index = load_dr1899_csv(Path(args.offline_path))
        for ref in refs:
            passage, missing = fetch_passage_offline(ref, verse_index=verse_index)
            passages[ref] = passage
            offline_missing[ref] = missing
            total_missing += missing
            total_requested += len(tokenize(passage.text))
    else:
        if not api_key or not bible_id:
            raise SystemExit("SCRIPTURE_API_KEY and BIBLE_ID must be set (env or CLI).")
        for ref in refs:
            passages[ref] = fetch_passage_from_api(ref, bible_id=bible_id, api_key=api_key)
            total_requested += len(tokenize(passages[ref].text))

    G, totals = build_graph(passages)
    all_tokens = []
    for p in passages.values():
        all_tokens.extend(tokenize(p.text))

    rosary_counts = rosary_resonance(all_tokens)

    payload = {
        "date": target_date,
        "references": refs,
        "passages": [asdict(p) for p in passages.values()],
        "top_words": totals.most_common(25),
        "rosary_resonance": rosary_counts,
        "sources": {
            "usccb_page": f"https://bible.usccb.org/daily-bible-readings?date={target_date}",
            "bible_api": "https://scripture.api.bible" if args.mode == "api" else None,
            "translation_id": bible_id if args.mode == "api" else "Douay-Rheims 1899 (offline)",
            "mode": args.mode,
        },
        "offline_missing_counts": offline_missing if args.mode == "offline" else {},
        "meta": {
            "built_at": datetime.utcnow().isoformat() + "Z",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "total_tokens": len(all_tokens),
            "total_refs": len(refs),
            "total_requested_tokens": total_requested,
            "total_missing_tokens": total_missing if args.mode == "offline" else 0,
        },
    }
    json_path = data_dir / "latest_payload.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plot_hotwords(totals, output_dir / "hotwords.png")
    plot_graph(G, output_dir / "context_graph.png")
    save_gexf(G, output_dir / "context_graph.gexf")

    print(f"[ok] {target_date} readings: {', '.join(refs)}")
    print(f"[ok] JSON: {json_path}")
    print(f"[ok] Viz: {output_dir / 'hotwords.png'}, {output_dir / 'context_graph.png'}")
    print(f"[ok] Graph: {output_dir / 'context_graph.gexf'}")


if __name__ == "__main__":
    main()
