"""Download the official COMP 472 competition data to the local data/ directory.

Requirements:
    python -m pip install kaggle
    Kaggle competition access and Kaggle API authentication

Run from the repository root:
    python download_data.py

This script does not upload Kaggle datasets or any private API key to GitHub.
"""
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COMPETITION = "comp-472-project-fall-2026-concordia"
REQUIRED = {
    "image_train.csv",
    "image_test.csv",
    "sentiment_train.csv",
    "sentiment_test.csv",
    "sample_submission.csv",
}

def main():
    DATA.mkdir(exist_ok=True)
    missing = REQUIRED - {p.name for p in DATA.glob("*.csv")}
    if not missing:
        print("All five Kaggle CSV files are already in data/.")
        return
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "competitions", "download",
             "-c", COMPETITION, "-p", str(DATA), "--force"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Kaggle download failed. Install the Kaggle package, sign in to Kaggle "
            "and accept the competition rules on the Kaggle website, then retry. "
            "Do not upload your kaggle.json or API key to GitHub."
        ) from exc
    archives = list(DATA.glob("*.zip"))
    if not archives:
        raise SystemExit("Kaggle command completed but no ZIP was found in data/.")
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename).name
                if name in REQUIRED:
                    target = DATA / name
                    if not target.exists() or name in missing:
                        with z.open(member) as src, target.open("wb") as dst:
                            import shutil
                            shutil.copyfileobj(src, dst)
    remaining = REQUIRED - {p.name for p in DATA.glob("*.csv")}
    if remaining:
        raise SystemExit(f"Missing CSV files after download: {sorted(remaining)}")
    print("Success: all five CSV files are available in data/.")

if __name__ == "__main__":
    main()
