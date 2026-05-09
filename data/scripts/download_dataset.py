"""
Download the LIAR dataset from the original source.

Usage:
    python data/scripts/download_dataset.py --dataset liar --output data/raw/liar

LIAR dataset paper:
  William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset
  for Fake News Detection, ACL 2017.
  https://www.cs.ucsb.edu/~william/papers/liar_dataset.pdf
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests

LIAR_URL = (
    "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
)

HEADERS = {
    "User-Agent": "FakeNewsDetector/1.0 (research use)"
}


def download_liar(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    zip_path = out / "liar_dataset.zip"

    if (out / "train.tsv").exists():
        print("LIAR dataset already downloaded.")
        return

    print(f"Downloading LIAR dataset → {zip_path}")
    try:
        resp = requests.get(LIAR_URL, headers=HEADERS, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)
        print()
    except requests.RequestException as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            "\nAlternative: download manually from "
            "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip "
            f"and extract to {output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Extracting to {out}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)

    zip_path.unlink()  # remove archive after extraction

    # Verify expected files
    expected = ["train.tsv", "valid.tsv", "test.tsv"]
    missing = [f for f in expected if not (out / f).exists()]
    if missing:
        print(f"Warning: expected files not found: {missing}", file=sys.stderr)
    else:
        print("LIAR dataset ready:")
        for f in expected:
            size = (out / f).stat().st_size // 1024
            print(f"  {out / f}  ({size} KB)")


def download_fakenewsnet_instructions() -> None:
    print(
        """
FakeNewsNet requires the official crawler:
  https://github.com/KaiDMML/FakeNewsNet

Steps:
  1. git clone https://github.com/KaiDMML/FakeNewsNet
  2. pip install -r requirements.txt
  3. python fakenewsnet.py --politifact fake real --gossipcop fake real

Then export a CSV with columns ['text', 'label'] to data/raw/fakenewsnet.csv
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download fake news datasets")
    parser.add_argument(
        "--dataset",
        choices=["liar", "fakenewsnet"],
        default="liar",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output",
        default="data/raw/liar",
        help="Output directory",
    )
    args = parser.parse_args()

    if args.dataset == "liar":
        download_liar(args.output)
    else:
        download_fakenewsnet_instructions()
