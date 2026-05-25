"""Download the trained TorchScript model from a GitHub release.

By default, fetches the v1.0.0 release asset
`distraction_classifier.pt` from github.com/adit-rah/lock-in and writes it to
models/distraction_classifier.pt. Override the URL with --url for testing.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://github.com/adit-rah/lock-in/releases/download/v1.0.0/"
    "distraction_classifier.pt"
)
DEFAULT_DEST = "models/distraction_classifier.pt"


def download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)

    if expected_sha256:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch: expected {expected_sha256}, got {actual}"
            )

    tmp.replace(dest)
    size_mb = dest.stat().st_size / (1 << 20)
    print(f"Wrote {dest} ({size_mb:.1f} MB)")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"Source URL (default: {DEFAULT_URL})")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"Output path (default: {DEFAULT_DEST})")
    parser.add_argument("--sha256", default=None,
                        help="Optional expected SHA-256 of the downloaded file")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the destination already exists")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.exists() and not args.force:
        print(f"{dest} already exists. Use --force to re-download.")
        return 0

    try:
        download(args.url, dest, expected_sha256=args.sha256)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
