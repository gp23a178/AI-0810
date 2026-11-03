# -*- coding: utf-8 -*-
"""天鳳ランキングの四人麻雀「牌譜 DL」ZIPを全件ダウンロードする。

標準ライブラリのみを使用します。初期設定では現在のランキングページに掲載された
四人麻雀のリンクを対象とします。三人麻雀のリンク(pf3)は対象外です。
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
import time
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

RANKING_URL = "https://tenhou.net/ranking.html"
BASE_DIR = Path("tenhou_data")
ARCHIVE_DIR = BASE_DIR / "archives"
MANIFEST_PATH = BASE_DIR / "download_manifest.csv"
DELAY_SECONDS = 1.0
RETRIES = 3
TIMEOUT_SECONDS = 120
ZIP_LINK_RE = re.compile(r"mjlog_pf4-[^/\\?#]*?n\d+\.zip", re.IGNORECASE)
DATE_RE = re.compile(r"(?P<y>20\d{2})[./-](?P<m>\d{2})[./-](?P<d>\d{2})")
BATTLES_RE = re.compile(r"([\d,]+)\s*戦\s*(.*)")


class RankingParser(HTMLParser):
    """<br>等で区切られたランキング行からテキストとリンクを取得する。"""

    BREAK_TAGS = {"br", "tr", "li", "p", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines = []
        self._parts = []
        self._links = []
        self._anchor = None

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._parts)).strip()
        if text or self._links:
            self.lines.append({"text": text, "links": list(self._links)})
        self._parts = []
        self._links = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.BREAK_TAGS:
            self._flush()
        if tag == "a":
            self._anchor = {"href": dict(attrs).get("href", ""), "text_parts": []}

    def handle_data(self, data):
        self._parts.append(data)
        if self._anchor is not None:
            self._anchor["text_parts"].append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "a" and self._anchor is not None:
            self._links.append({
                "href": self._anchor.get("href", ""),
                "text": re.sub(r"\s+", " ", "".join(self._anchor["text_parts"])).strip(),
            })
            self._anchor = None
        if tag in self.BREAK_TAGS:
            self._flush()

    def close(self):
        super().close()
        self._flush()


def fetch_page(url: str) -> str:
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; TenhouArchiveDownloader/1.0)",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    })
    with urlopen(req, timeout=30) as response:
        encoding = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(encoding, errors="replace")


def normalize_date(text: str) -> str:
    m = DATE_RE.search(text)
    if not m:
        return ""
    return f"{m.group('y')}-{m.group('m')}-{m.group('d')}"


def clean_player(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" |｜:：\t\r\n　")
    # Last row may be followed by another page section if HTML omitted a line break.
    for marker in ("■段位戦", "■", "© C-EGG", "UPDATED"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text.strip(" |｜:：\t\r\n　")


def parse_ranking_entries(html: str):
    parser = RankingParser()
    parser.feed(html)
    parser.close()
    entries = []
    seen = set()

    for line in parser.lines:
        candidates = []
        for link in line["links"]:
            href = link.get("href", "")
            url = urljoin(RANKING_URL, href)
            path_name = urlparse(url).path.rsplit("/", 1)[-1]
            if not ZIP_LINK_RE.search(path_name):
                continue
            if url in seen:
                continue
            seen.add(url)
            candidates.append((url, path_name))
        if not candidates:
            continue

        text = line["text"]
        date = normalize_date(text)
        battle_match = BATTLES_RE.search(text)
        battles = battle_match.group(1).replace(",", "") if battle_match else ""
        player = clean_player(battle_match.group(2)) if battle_match else ""
        if not player:
            # Do not guess from arbitrary text: use the archive filename as an explicit fallback.
            player = "unknown_" + candidates[0][1].replace(".zip", "")

        for url, filename in candidates:
            entries.append({
                "player": player,
                "achievement_date": date,
                "battles_at_listing": battles,
                "zip_url": url,
                "zip_filename": filename,
                "source_page": RANKING_URL,
                "local_path": str((ARCHIVE_DIR / filename).as_posix()),
                "status": "pending",
                "size_bytes": "",
                "sha256": "",
                "error": "" if date and not player.startswith("unknown_") else
                         "ランキング行から日付またはプレイヤー名を完全に取得できませんでした。manifestを確認してください。",
            })

    # Guard against mistakenly scraping a page that no longer uses the documented ZIP links.
    if not entries:
        raise RuntimeError(
            "四人麻雀のZIPリンク(mjlog_pf4-...zip)が見つかりませんでした。\n"
            "ランキングページの構造やリンク形式が変わった可能性があります。"
        )
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad_member = zf.testzip()
            return bad_member is None
    except (OSError, zipfile.BadZipFile, EOFError):
        return False


def download_one(entry: dict) -> dict:
    filename = entry["zip_filename"]
    destination = ARCHIVE_DIR / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and is_valid_zip(destination):
        entry["status"] = "already_downloaded"
        entry["size_bytes"] = str(destination.stat().st_size)
        entry["sha256"] = sha256_file(destination)
        return entry

    temp_path = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            if temp_path.exists():
                temp_path.unlink()
            req = Request(entry["zip_url"], headers={
                "User-Agent": "Mozilla/5.0 (compatible; TenhouArchiveDownloader/1.0)",
                "Referer": RANKING_URL,
                "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
            })
            with urlopen(req, timeout=TIMEOUT_SECONDS) as response, temp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if not is_valid_zip(temp_path):
                raise RuntimeError("ダウンロード結果が正常なZIPファイルではありません。")
            temp_path.replace(destination)
            entry["status"] = "downloaded"
            entry["size_bytes"] = str(destination.stat().st_size)
            entry["sha256"] = sha256_file(destination)
            entry["error"] = "" if not entry.get("error") else entry["error"]
            return entry
        except Exception as exc:
            last_error = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            if attempt < RETRIES:
                wait = 2 * attempt
                print(f"  取得エラー ({attempt}/{RETRIES}): {exc} — {wait}秒後に再試行")
                time.sleep(wait)
    entry["status"] = "failed"
    entry["error"] = f"{entry.get('error', '')} ダウンロード失敗: {last_error}".strip()
    return entry


def write_manifest(entries):
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "player", "achievement_date", "battles_at_listing", "zip_url",
        "zip_filename", "source_page", "local_path", "status", "size_bytes",
        "sha256", "error",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"公式ランキングを取得します: {RANKING_URL}")
    html = fetch_page(RANKING_URL)
    entries = parse_ranking_entries(html)
    print(f"四人麻雀のZIPリンクを {len(entries)} 件検出しました。")
    print("三人麻雀(pf3)のリンクは対象外です。")

    # Save a pending manifest first so an interrupted run is auditable.
    write_manifest(entries)
    for i, entry in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] {entry['player']} — {entry['zip_filename']}")
        result = download_one(entry)
        print(f"  状態: {result['status']}")
        if result.get("size_bytes"):
            print(f"  サイズ: {int(result['size_bytes']) / (1024 * 1024):.1f} MB")
        if result.get("error"):
            print(f"  注意: {result['error']}")
        write_manifest(entries)  # checkpoint after every archive
        time.sleep(DELAY_SECONDS)

    success = sum(e["status"] in {"downloaded", "already_downloaded"} for e in entries)
    failed = len(entries) - success
    print("\nダウンロード処理が終了しました。")
    print(f"正常なZIP: {success} / {len(entries)}、失敗: {failed}")
    print(f"ZIP保存先: {ARCHIVE_DIR.resolve()}")
    print(f"一覧・天鳳位達成日: {MANIFEST_PATH.resolve()}")
    if any(not e.get("achievement_date") or e.get("player", "").startswith("unknown_") for e in entries):
        print("注意: manifestに日付/名前不明の行があります。変換コードはその行の牌譜を安全側で除外します。")
    if failed:
        print("失敗分は同じコマンドを再実行すると、有効なZIPをスキップして再試行します。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断されました。保存済みZIPとmanifestは保持されています。")
        sys.exit(130)
    except Exception as exc:
        print("\n処理を続行できませんでした:")
        print(exc)
        sys.exit(1)
