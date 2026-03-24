"""Walk the project data directory and load markdown files with document-level metadata."""

from __future__ import annotations

from pathlib import Path

from .models import ChunkMetadata, DocumentType

# Pre-chunk full document; chunking step will replace with real strategy names.
DOCUMENT_LEVEL_STRATEGY = "document"

_FOLDER_TO_TYPE: dict[str, DocumentType] = {
    "city_guides":     "city_guide",
    "visa_info":       "visa_info",
    "coworking":       "coworking_review",
    "cost_comparison": "cost_comparison",
}

_FILE_TO_GEO: dict[str, tuple[str | None, str | None]] = {
    "medellin_colombia":                    ("Medellín",    "Colombia"),
    "florianopolis_brazil":                 ("Florianópolis","Brazil"),
    "mexico_city_mexico":                   ("Mexico City", "Mexico"),
    "buenos_aires_argentina":               ("Buenos Aires","Argentina"),
    "latin_america_dnv_summary_table":      (None,          None),
    "argentina_visa_options_bullets":       (None,          "Argentina"),
    "mexico_visa_paths_overview":           (None,          "Mexico"),
    "medellin_florianopolis_reviews":       (None,          None),
    "mexico_city_buenos_aires_reviews":     (None,          None),
    "medellin_vs_mexico_city":              (None,          None),
    "four_city_rent_and_coworking_snapshot":(None,          None),
}

def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"


def _build_metadata(path: Path, root: Path) -> ChunkMetadata:
    """ Build ChunkMetadata from a file path. """
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"Expected category subfolder under data/: {rel}")

    stem = path.stem
    city, country = _FILE_TO_GEO.get(stem, (None, None))

    folder = parts[0]
    document_type = _FOLDER_TO_TYPE.get(folder)

    if document_type is None:
        raise ValueError(f"Unknown folder type: {folder}")

    return ChunkMetadata(
        source_file=str(rel),
        document_type=document_type,
        country=country,
        city=city,
        section=None,
        chunk_strategy=DOCUMENT_LEVEL_STRATEGY,
    )

def loader(data_root: Path | None = None) -> list[tuple[str, ChunkMetadata]]:
    """
    Walk ``data_root`` (default: repo ``data/``) and return a list of
    ``(raw_text, ChunkMetadata)`` for each ``*.md`` file. Paths are sorted for stable ordering.

    ``document_type`` is inferred from the folder name (e.g. ``city_guides`` → ``city_guide``);
    see ``_FOLDER_TO_TYPE``.
    """
    root = data_root if data_root is not None else _default_data_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    results = []
    for path in sorted(root.rglob("*.md")):
        text, metadata = load_file(path, root)
        results.append((text, metadata))

    return results

def load_file(path: Path, root: Path | None = None) -> tuple[str, ChunkMetadata]:
    """Load a single .md file and return (text, ChunkMetadata)."""
    root = root if root is not None else _default_data_dir()
    return path.read_text(encoding="utf-8"), _build_metadata(path, root)
