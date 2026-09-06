"""
事前学習用コーパスを作るスクリプト。

2つの収集元に対応:
    - Wikipedia日本語版 (MediaWiki API): 幅広い分野の百科事典的な文章
    - 青空文庫 (公式カタログCSV): 著作権切れの文学作品(小説・随筆など)

集めたテキストは、モデルに「日本語としての自然な文法・語彙」を
事前学習(pretrain)させるために使う。また、どのデータをどこから集めたかを
data/sources.csv (生ログ) と data/SOURCES.md (集計・一覧表) に記録する。

実行例:
    python scripts/build_corpus.py --source wikipedia --target_chars 1000000
    python scripts/build_corpus.py --source aozora --target_chars 1000000
    python scripts/build_corpus.py --source both --target_chars 1000000
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

WIKI_API_URL = "https://ja.wikipedia.org/w/api.php"
AOZORA_CATALOG_URL = "http://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
HEADERS = {"User-Agent": "chatai-corpus-builder/1.0 (educational project; contact: local-user)"}

ROOT = Path(__file__).parent.parent
CATALOG_CACHE = ROOT / "data" / ".cache" / "aozora_list_person_all_extended_utf8.csv"
SOURCES_CSV = ROOT / "data" / "sources.csv"
SOURCES_MD = ROOT / "data" / "SOURCES.md"
SOURCE_FIELDS = ["source_type", "title", "url", "char_count", "collected_at"]

RUBY_PATTERN = re.compile(r"《[^》]*》")
NOTE_PATTERN = re.compile(r"［＃[^］]*］")
SEPARATOR = "-------------------------------------------------------"


# ---------------------------------------------------------------------------
# 収集元データ表 (どのデータをどこから集めたかの記録)
# ---------------------------------------------------------------------------

def append_source(source_type: str, title: str, url: str, char_count: int) -> None:
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not SOURCES_CSV.exists()
    with SOURCES_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "source_type": source_type,
                "title": title,
                "url": url,
                "char_count": char_count,
                "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )


def regenerate_sources_markdown() -> None:
    if not SOURCES_CSV.exists():
        return
    with SOURCES_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total_chars = sum(int(r["char_count"]) for r in rows)
    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        entry = by_type.setdefault(r["source_type"], {"count": 0, "chars": 0})
        entry["count"] += 1
        entry["chars"] += int(r["char_count"])

    lines = [
        "# 学習データの収集元一覧",
        "",
        "このファイルは `scripts/build_corpus.py` によって自動生成されます。",
        "",
        f"総件数: {len(rows):,} / 総文字数: {total_chars:,}",
        "",
        "## ソース別集計",
        "",
        "| 収集元 | 件数 | 文字数 |",
        "|---|---|---|",
    ]
    for k, v in sorted(by_type.items()):
        lines.append(f"| {k} | {v['count']:,} | {v['chars']:,} |")

    lines += ["", "## 収集した個別データ一覧", "", "| 収集元 | タイトル | URL | 文字数 | 収集日時(UTC) |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['source_type']} | {r['title']} | {r['url']} | {int(r['char_count']):,} | {r['collected_at']} |"
        )

    SOURCES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Wikipedia日本語版からの収集
# ---------------------------------------------------------------------------

def fetch_wikipedia_batch(batch_size: int) -> list[dict]:
    """ランダムなgrnlimit件の記事(タイトル・本文プレーンテキスト)を1回のAPI呼び出しで取得する。"""
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": batch_size,
        "prop": "extracts",
        "explaintext": 1,
    }
    r = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    return [{"title": p.get("title", ""), "extract": p.get("extract", "")} for p in pages.values()]


def collect_wikipedia(out_file, target_chars: int, batch_size: int, min_len: int, sleep: float) -> int:
    total = 0
    seen_titles: set[str] = set()

    while total < target_chars:
        try:
            articles = fetch_wikipedia_batch(batch_size)
        except requests.RequestException as e:
            is_rate_limited = getattr(e, "response", None) is not None and e.response.status_code == 429
            wait = 10 if is_rate_limited else 2
            print(f"[wikipedia] リクエスト失敗、{wait}秒待ってリトライします: {e}")
            time.sleep(wait)
            continue

        for article in articles:
            text = article["extract"].strip()
            title = article["title"]
            if len(text) < min_len or title in seen_titles:
                continue
            seen_titles.add(title)

            out_file.write(text)
            out_file.write("\n\n")
            total += len(text)

            url = "https://ja.wikipedia.org/wiki/" + title.replace(" ", "_")
            append_source("wikipedia", title, url, len(text))

        print(f"[wikipedia] 収集済み: {total:,} / {target_chars:,} 文字")
        time.sleep(sleep)

    return total


# ---------------------------------------------------------------------------
# 青空文庫からの収集 (著作権切れ作品のみ)
# ---------------------------------------------------------------------------

def load_aozora_catalog() -> list[dict]:
    if not CATALOG_CACHE.exists():
        CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("[aozora] 公式カタログをダウンロード中...")
        r = requests.get(AOZORA_CATALOG_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            raw = zf.read(csv_name)
        text = raw.decode("utf-8-sig", errors="replace")
        CATALOG_CACHE.write_text(text, encoding="utf-8")
    else:
        text = CATALOG_CACHE.read_text(encoding="utf-8")

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        url = row.get("テキストファイルURL", "")
        if not url.startswith("https://www.aozora.gr.jp") or not url.endswith(".zip"):
            continue
        if row.get("作品著作権フラグ") != "なし":
            continue
        rows.append(row)
    return rows


def clean_aozora_text(raw: str) -> str:
    parts = raw.split(SEPARATOR)
    body = parts[2] if len(parts) >= 3 else raw
    body = body.split("底本：")[0]
    body = RUBY_PATTERN.sub("", body)
    body = NOTE_PATTERN.sub("", body)
    body = body.replace("｜", "")
    return body.strip()


def fetch_aozora_text(zip_url: str) -> str:
    r = requests.get(zip_url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        txt_name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        raw_bytes = zf.read(txt_name)
    try:
        raw = raw_bytes.decode("shift_jis")
    except UnicodeDecodeError:
        raw = raw_bytes.decode("cp932", errors="replace")
    return clean_aozora_text(raw)


def collect_aozora(out_file, target_chars: int, min_len: int, sleep: float) -> int:
    catalog = load_aozora_catalog()
    random.shuffle(catalog)
    print(f"[aozora] 著作権切れ作品 {len(catalog):,} 件からランダムに収集します。")

    total = 0
    for row in catalog:
        if total >= target_chars:
            break

        title = row.get("作品名", "")
        author = f"{row.get('姓', '')}{row.get('名', '')}"
        url = row["テキストファイルURL"]

        try:
            text = fetch_aozora_text(url)
        except Exception as e:
            print(f"[aozora] 取得失敗、スキップ: {title} ({e})")
            continue

        if len(text) < min_len:
            continue

        out_file.write(text)
        out_file.write("\n\n")
        total += len(text)

        append_source("aozora", f"{title}({author})", url, len(text))
        print(f"[aozora] {title}: {len(text):,}文字 (累計 {total:,} / {target_chars:,})")
        time.sleep(sleep)

    return total


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["wikipedia", "aozora", "both"], default="wikipedia")
    parser.add_argument("--out", default="data/pretrain_corpus.txt")
    parser.add_argument("--target_chars", type=int, default=1_000_000, help="収集する目標文字数")
    parser.add_argument("--batch_size", type=int, default=500, help="Wikipedia: 1回のAPI呼び出しで取得する記事数(未認証ユーザーの上限500)")
    parser.add_argument("--min_len", type=int, default=200, help="この文字数未満のテキストは捨てる")
    parser.add_argument("--sleep", type=float, default=0.5, help="APIへの負荷軽減のためのリクエスト間隔(秒)")
    parser.add_argument("--append", action="store_true", help="既存のoutファイルに追記する(既定は上書き)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    with out_path.open(mode, encoding="utf-8") as f:
        total = 0
        if args.source == "wikipedia":
            total += collect_wikipedia(f, args.target_chars, args.batch_size, args.min_len, args.sleep)
        elif args.source == "aozora":
            total += collect_aozora(f, args.target_chars, args.min_len, args.sleep)
        else:  # both: 半分ずつ配分する
            half = args.target_chars // 2
            total += collect_wikipedia(f, half, args.batch_size, args.min_len, args.sleep)
            total += collect_aozora(f, args.target_chars - half, args.min_len, args.sleep)

    regenerate_sources_markdown()
    print(f"完了: {out_path} に約 {total:,} 文字を保存しました。")
    print(f"収集元一覧: {SOURCES_MD}")


if __name__ == "__main__":
    main()
