"""
Parsing des guidelines PDF en texte structuré.

Pour chaque PDF dans raw_pdfs/ :
1. Extraction du texte page par page (PyMuPDF)
2. Nettoyage (headers/footers répétitifs, espaces multiples, artefacts)
3. Sauvegarde en JSON dans ingestion/guidelines/processed/

Usage:
    python -m ingestion.guidelines.parse
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

from ingestion.guidelines.sources import GUIDELINES

RAW_DIR = Path(__file__).parent / "raw_pdfs"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Lignes à supprimer si elles apparaissent identiques sur plusieurs pages
# (headers/footers répétitifs typiques des PDF institutionnels)
MIN_REPEAT_TO_STRIP = 3


def extract_pages(pdf_path: Path) -> list[str]:
    """Extrait le texte brut de chaque page d'un PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def find_repeated_lines(pages: list[str]) -> set[str]:
    """
    Identifie les lignes qui se répètent sur plusieurs pages
    (ex: 'WHO/UCN/NCD/20.1', numéros de page, copyright en footer).
    Ces lignes n'apportent aucune information utile au RAG.
    """
    line_counts: dict[str, int] = {}
    for page_text in pages:
        for line in page_text.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) < 100:  # on ignore les vraies phrases longues
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

    return {line for line, count in line_counts.items() if count >= MIN_REPEAT_TO_STRIP}


def clean_page_text(text: str, repeated_lines: set[str]) -> str:
    """Nettoie le texte d'une page : retire headers/footers et normalise les espaces."""
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if line.strip() not in repeated_lines]
    cleaned = "\n".join(cleaned_lines)

    # Normalisation des espaces et sauts de ligne multiples
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Numéros de page isolés (ex: une ligne avec juste "42")
    cleaned = re.sub(r"\n\d{1,4}\n", "\n", cleaned)

    return cleaned.strip()


def parse_guideline(pdf_path: Path) -> dict:
    """Parse un PDF complet en structure {pages: [...], full_text: ...}."""
    raw_pages = extract_pages(pdf_path)
    repeated_lines = find_repeated_lines(raw_pages)

    cleaned_pages = []
    for i, raw_text in enumerate(raw_pages, start=1):
        cleaned = clean_page_text(raw_text, repeated_lines)
        if cleaned:  # on ignore les pages vides (couvertures, pages blanches)
            cleaned_pages.append({"page_number": i, "text": cleaned})

    full_text = "\n\n".join(p["text"] for p in cleaned_pages)

    return {
        "pages": cleaned_pages,
        "full_text": full_text,
        "total_pages": len(raw_pages),
        "usable_pages": len(cleaned_pages),
    }


def main():
    print(f"📄 Parsing de {len(GUIDELINES)} guidelines\n")

    summary = []

    for guideline in GUIDELINES:
        name = guideline["name"]
        pdf_path = RAW_DIR / f"{name}.pdf"

        if not pdf_path.exists():
            print(f"⚠️  {name} : PDF introuvable, on passe\n")
            continue

        print(f"📂 {guideline['title']}")
        parsed = parse_guideline(pdf_path)

        output = {
            "name": name,
            "title": guideline["title"],
            "topic": guideline["topic"],
            "source_url": guideline["url"],
            **parsed,
        }

        output_path = PROCESSED_DIR / f"{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        char_count = len(parsed["full_text"])
        print(
            f"   ✅ {parsed['usable_pages']}/{parsed['total_pages']} pages exploitables, "
            f"{char_count:,} caractères → {output_path}\n"
        )

        summary.append(
            {
                "name": name,
                "pages": parsed["usable_pages"],
                "chars": char_count,
            }
        )

    total_chars = sum(s["chars"] for s in summary)
    print(f"{'='*50}")
    print(
        f"✅ TERMINÉ — {len(summary)} guidelines parsées, {total_chars:,} caractères au total"
    )
    print(f"{'='*50}")
    for s in summary:
        print(f"  {s['name']:40s} {s['pages']:3d} pages  {s['chars']:>8,} car.")


if __name__ == "__main__":
    main()
