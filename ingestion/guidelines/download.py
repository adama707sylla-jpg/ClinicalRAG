"""
Téléchargement des guidelines OMS en PDF.

Usage:
    python -m ingestion.guidelines.download
"""

from pathlib import Path

import requests
from tqdm import tqdm

from ingestion.guidelines.sources import GUIDELINES

RAW_DIR = Path(__file__).parent / "raw_pdfs"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ClinicalRAG-bot/1.0)"}


def download_pdf(url: str, dest_path: Path) -> bool:
    try:
        response = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=dest_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except requests.RequestException as e:
        print(f"   ❌ Échec: {e}")
        return False


def main():
    print(f"📥 Téléchargement de {len(GUIDELINES)} guidelines OMS\n")

    success_count = 0
    for guideline in GUIDELINES:
        name = guideline["name"]
        url = guideline["url"]
        dest_path = RAW_DIR / f"{name}.pdf"

        if dest_path.exists():
            print(f"⏭️  {name} déjà téléchargé, on passe")
            success_count += 1
            continue

        print(f"📄 {guideline['title']}")
        if download_pdf(url, dest_path):
            print(f"   ✅ Sauvegardé → {dest_path}\n")
            success_count += 1
        else:
            print(f"   ⚠️  Tu devras le télécharger manuellement: {url}\n")

    print(f"{'='*50}")
    print(f"✅ {success_count}/{len(GUIDELINES)} guidelines téléchargées")


if __name__ == "__main__":
    main()
