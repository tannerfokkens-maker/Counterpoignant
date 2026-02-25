#!/usr/bin/env python3
"""Download .krn files from KernScores for Bach and related polyphonic composers.

Fetches search result pages, extracts .krn download URLs, deduplicates by
filename per composer, and downloads into data/midi/kernscores/<composer>/.

Resumable — skips files already downloaded.
"""

import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "https://kern.humdrum.org/cgi-bin/ksdata"
SEARCH = "https://kern.humdrum.org/search?s=t&keyword="
OUT = Path("data/midi/kernscores")
CACHE = Path("/tmp/kernscores_cache")

# (search_keyword, output_dirname)
COMPOSERS = [
    ("Bach+Johann", "bach"),
    ("Buxtehude", "buxtehude"),
    ("Pachelbel", "pachelbel"),
    ("Corelli", "corelli"),
    ("Vivaldi", "vivaldi"),
    ("Frescobaldi", "frescobaldi"),
    ("Josquin", "josquin"),
    ("Victoria", "victoria"),
    ("Lassus", "lassus"),
    ("Byrd", "byrd"),
    ("Haydn", "haydn"),
    ("Mozart", "mozart"),
    ("Beethoven", "beethoven"),
    ("Dufay", "dufay"),
    ("Dunstable", "dunstable"),
    ("Isaac", "isaac"),
    ("Monteverdi", "monteverdi"),
    ("Banchieri", "banchieri"),
    ("Giovannelli", "giovannelli"),
    ("Vecchi", "vecchi"),
    ("Clementi", "clementi"),
    ("Scarlatti", "scarlatti"),
]

PAIR_RE = re.compile(r'location=([^&]+)&file=([^&]+\.krn)')


def fetch_url(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    total_new = 0
    total_skip = 0
    total_fail = 0

    for keyword, dirname in COMPOSERS:
        print()
        print("=" * 55)
        print(f"  {dirname} (search: {keyword})")
        print("=" * 55)

        html_file = CACHE / f"{dirname}.html"

        # Fetch search page (cached)
        if not html_file.exists() or html_file.stat().st_size < 100:
            print("  Fetching search results...")
            if fetch_url(f"{SEARCH}{keyword}", html_file):
                print(f"  Saved to {html_file}")
            else:
                print(f"  WARNING: Could not fetch search page, skipping")
                continue
            time.sleep(0.5)
        else:
            print(f"  Using cached search results")

        html = html_file.read_text(errors="replace")

        # Extract location+file pairs
        pairs = set(PAIR_RE.findall(html))
        if not pairs:
            print("  No .krn files found, skipping")
            continue
        print(f"  Found {len(pairs)} location+file pairs")

        # Deduplicate by filename (keep first occurrence)
        seen = set()
        download_list = []
        for location, filename in sorted(pairs):
            if filename not in seen:
                seen.add(filename)
                download_list.append((location, filename))

        print(f"  Deduplicated to {len(download_list)} unique files")

        # Download
        comp_dir = OUT / dirname
        comp_dir.mkdir(parents=True, exist_ok=True)
        new = 0
        skip = 0
        fail = 0

        for location, filename in download_list:
            outfile = comp_dir / filename
            if outfile.exists():
                skip += 1
                continue

            url = f"{BASE}?location={location}&file={filename}&format=kern"
            if fetch_url(url, outfile):
                new += 1
            else:
                fail += 1
                outfile.unlink(missing_ok=True)

            time.sleep(0.1)

        actual = len(list(comp_dir.glob("*.krn")))
        print(f"  Result: {new} new, {skip} already had, {fail} failed ({actual} total on disk)")

        total_new += new
        total_skip += skip
        total_fail += fail

    # Summary
    total_files = len(list(OUT.rglob("*.krn")))
    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  New downloads: {total_new}")
    print(f"  Already had:   {total_skip}")
    print(f"  Failed:        {total_fail}")
    print(f"  Total .krn files on disk: {total_files}")
    print()
    print(f"  Files saved to {OUT}/")
    print("  Run 'bach-gen prepare-data' to include in training.")


if __name__ == "__main__":
    main()
