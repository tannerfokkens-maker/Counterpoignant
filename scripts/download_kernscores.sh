#!/usr/bin/env python3
"""Download .krn files from KernScores for Bach and related polyphonic composers.

Fetches search result pages, extracts .krn download URLs, deduplicates by
filename per composer, and downloads into data/midi/kernscores/<composer>/.

Resumable — skips files already downloaded.
"""

import os
import re
import ssl
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

BASE = "https://kern.humdrum.org/cgi-bin/ksdata"
SEARCH = "https://kern.humdrum.org/search?s=t&keyword="
OUT = Path("data/midi/kernscores")
CACHE = Path("/tmp/kernscores_cache")

# (search_keyword, output_dirname)
COMPOSERS = [
    # Existing corpus
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

    # Additional KernScores composers currently on-site
    ("Adam+Adolphe", "adam"),
    ("Alkan", "alkan"),
    ("Billings+William", "billings"),
    ("Bossi", "bossi"),
    ("Brahms", "brahms"),
    ("Chopin", "chopin"),
    ("Field+John", "field"),
    ("Flecha", "flecha"),
    ("Foster+Stephen", "foster"),
    ("Gershwin", "gershwin"),
    ("Grieg", "grieg"),
    ("Himmel", "himmel"),
    ("Hummel", "hummel"),
    ("Ives", "ives"),
    ("Joplin", "joplin"),
    ("Landini", "landini"),
    ("Liszt", "liszt"),
    ("MacDowell", "macdowell"),
    ("Mendelssohn", "mendelssohn"),
    ("Prokofiev", "prokofiev"),
    ("Ravel", "ravel"),
    ("Schubert", "schubert"),
    ("Schumann", "schumann"),
    ("Scriabin", "scriabin"),
    ("Sinding", "sinding"),
    ("Sousa", "sousa"),
    ("Turpin", "turpin"),
    ("Weber+Carl", "weber"),
]

PAIR_RE = re.compile(r'location=([^&]+)&file=([^&]+\.krn)')
INVALID_NAME_RE = re.compile(
    r"(?:-auto|-combined|-beat|-sampled|-pan|-nopan|extractf|-20|-60|-80|-S)(?:\.krn)$",
    re.IGNORECASE,
)
REJECT_NAME_RE_BY_DIR: dict[str, re.Pattern[str]] = {
    # KernScores search can return non-Bach material for the Bach keyword.
    # These filename families are persistent non-Bach intruders in this corpus.
    "bach": re.compile(r"^(?:xxerk\d+|piston\d+)\.krn$", re.IGNORECASE),
}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}
SSL_CTX: ssl.SSLContext | None = None
SSL_VERIFY_ERROR_HINT = (
    "SSL certificate verification failed. Install/update certificate roots "
    "(e.g., `uv add certifi`) or set KERNSCORES_INSECURE_SSL=1 as a last resort."
)


def _build_ssl_context() -> ssl.SSLContext | None:
    """Return SSL context with robust CA roots when possible."""
    insecure = os.environ.get("KERNSCORES_INSECURE_SSL", "").strip().lower()
    if insecure in {"1", "true", "yes"}:
        return ssl._create_unverified_context()

    # Prefer certifi when installed in this environment.
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_url(
    url: str,
    dest: Path,
    retries: int = 3,
    timeout_sec: float = 20.0,
) -> tuple[bool, str]:
    """Fetch URL to disk with retries; return (success, error_message)."""
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=REQUEST_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout_sec, context=SSL_CTX) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True, ""
        except urllib.error.HTTPError as exc:
            last_err = f"HTTPError {exc.code}: {exc.reason}"
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            if "CERTIFICATE_VERIFY_FAILED" in reason:
                last_err = f"URLError: {reason}. {SSL_VERIFY_ERROR_HINT}"
            else:
                last_err = f"URLError: {reason}"
        except OSError as exc:
            last_err = f"OSError: {exc}"

        if attempt < retries:
            time.sleep(min(0.5 * attempt, 2.0))

    return False, last_err or "unknown error"


def is_valid_krn(path: Path) -> bool:
    """Quick sanity checks to reject HTML error pages saved as .krn."""
    try:
        if not path.exists() or path.stat().st_size < 32:
            return False
        text = path.read_text(errors="replace")
    except OSError:
        return False

    head = text[:4096].lower()
    if "<html" in head or "access unsuccessful" in head:
        return False

    # Humdrum data should include at least one **kern exclusive interpretation.
    return "**kern" in text[:16384]


def should_skip_filename(dirname: str, filename: str) -> bool:
    if INVALID_NAME_RE.search(filename):
        return True
    reject_re = REJECT_NAME_RE_BY_DIR.get(dirname)
    if reject_re and reject_re.search(filename):
        return True
    return False


def main():
    global SSL_CTX
    SSL_CTX = _build_ssl_context()

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    total_new = 0
    total_skip = 0
    total_fail = 0
    total_invalid_existing = 0
    total_pruned_existing = 0

    for keyword, dirname in COMPOSERS:
        print()
        print("=" * 55)
        print(f"  {dirname} (search: {keyword})")
        print("=" * 55)

        html_file = CACHE / f"{dirname}.html"

        # Fetch search page (cached)
        if not html_file.exists() or html_file.stat().st_size < 100:
            print("  Fetching search results...")
            ok, err = fetch_url(f"{SEARCH}{keyword}", html_file, retries=3, timeout_sec=20.0)
            if ok:
                print(f"  Saved to {html_file}")
            else:
                # Keep moving if stale cache exists.
                if html_file.exists() and html_file.stat().st_size >= 100:
                    print(f"  WARNING: Search fetch failed ({err}); using cached file")
                else:
                    print(f"  WARNING: Could not fetch search page ({err}), skipping")
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

        # Group candidate locations per filename so dead links can fall back.
        candidates_by_file: dict[str, list[str]] = defaultdict(list)
        for location, filename in sorted(pairs):
            if should_skip_filename(dirname, filename):
                continue
            if location not in candidates_by_file[filename]:
                candidates_by_file[filename].append(location)

        print(f"  Candidate files after filter: {len(candidates_by_file)}")

        # Download
        comp_dir = OUT / dirname
        comp_dir.mkdir(parents=True, exist_ok=True)
        new = 0
        skip = 0
        fail = 0
        invalid_existing = 0
        pruned_existing = 0

        # Prune stale local junk so old bad files don't poison parsing later.
        for existing in comp_dir.glob("*.krn"):
            if should_skip_filename(dirname, existing.name) or not is_valid_krn(existing):
                existing.unlink(missing_ok=True)
                pruned_existing += 1

        for filename in sorted(candidates_by_file):
            outfile = comp_dir / filename
            if outfile.exists():
                if is_valid_krn(outfile):
                    skip += 1
                    continue
                invalid_existing += 1
                outfile.unlink(missing_ok=True)

            success = False
            tmpfile = outfile.with_suffix(".tmp")
            for location in candidates_by_file[filename]:
                url = f"{BASE}?location={location}&file={filename}&format=kern"
                ok, _err = fetch_url(url, tmpfile, retries=2, timeout_sec=20.0)
                if not ok:
                    continue
                if is_valid_krn(tmpfile):
                    tmpfile.replace(outfile)
                    new += 1
                    success = True
                    break
                tmpfile.unlink(missing_ok=True)
                time.sleep(0.05)

            if not success:
                fail += 1
                outfile.unlink(missing_ok=True)

            time.sleep(0.05)

        actual = len(list(comp_dir.glob("*.krn")))
        print(
            f"  Result: {new} new, {skip} already valid, {invalid_existing} invalid replaced, "
            f"{pruned_existing} stale pruned, "
            f"{fail} failed ({actual} total on disk)"
        )

        total_new += new
        total_skip += skip
        total_fail += fail
        total_invalid_existing += invalid_existing
        total_pruned_existing += pruned_existing

    # Summary
    total_files = len(list(OUT.rglob("*.krn")))
    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  New downloads: {total_new}")
    print(f"  Already valid: {total_skip}")
    print(f"  Invalid replaced: {total_invalid_existing}")
    print(f"  Stale pruned: {total_pruned_existing}")
    print(f"  Failed:        {total_fail}")
    print(f"  Total .krn files on disk: {total_files}")
    print()
    print(f"  Files saved to {OUT}/")
    print("  Run 'bach-gen prepare-data' to include in training.")


if __name__ == "__main__":
    main()
