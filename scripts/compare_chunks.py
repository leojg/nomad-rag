#!/usr/bin/env python3
"""Compare chunking strategies on a single markdown document or a batch by document type.

Examples:
  venv/bin/python scripts/compare_chunks.py data/city_guides/medellin_colombia.md
  venv/bin/python scripts/compare_chunks.py data/city_guides/medellin_colombia.md --output reports/
  venv/bin/python scripts/compare_chunks.py --type city_guide
  venv/bin/python scripts/compare_chunks.py --type visa_info --output reports/
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from ingestion.chunking import (
    ChunkingStrategy,
    MarkdownHeaderTextSplitterStrategy,
    RecursiveCharacterTextSplitterStrategy,
    SemanticChunkerStrategy,
)
from ingestion.loader import load_file, loader
from ingestion.models import ChunkMetadata, DocumentType

load_dotenv(ROOT / ".env")

DEFAULT_OUTPUT_DIR = ROOT / "reports"

_TYPE_TO_FOLDER: dict[str, str] = {
    "city_guide":       "city_guides",
    "visa_info":        "visa_info",
    "coworking_review": "coworking",
    "cost_comparison":  "cost_comparison",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    strategy_id: str
    chunks: list[tuple[str, ChunkMetadata]]

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def sizes(self) -> list[int]:
        return [len(text) for text, _ in self.chunks]

    @property
    def avg_size(self) -> int:
        return int(sum(self.sizes) / self.count) if self.count else 0

    @property
    def min_size(self) -> int:
        return min(self.sizes) if self.sizes else 0

    @property
    def max_size(self) -> int:
        return max(self.sizes) if self.sizes else 0


@dataclass
class DocumentResult:
    path: Path
    metadata: ChunkMetadata
    strategy_results: list[StrategyResult]


# ---------------------------------------------------------------------------
# Strategy instantiation
# ---------------------------------------------------------------------------

def build_strategies() -> list[ChunkingStrategy]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return [
        RecursiveCharacterTextSplitterStrategy(chunk_size=500, chunk_overlap=50),
        MarkdownHeaderTextSplitterStrategy(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        ),
        SemanticChunkerStrategy(embeddings=embeddings),
    ]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    strategies: list[ChunkingStrategy],
) -> DocumentResult:
    text, metadata = load_file(path)
    strategy_results = []
    for strategy in strategies:
        print(f"    [{strategy.id}]...", flush=True)
        chunks = strategy.chunk(text, metadata)
        # Filter empty chunks
        chunks = [(t, m) for t, m in chunks if t.strip()]
        strategy_results.append(StrategyResult(strategy_id=strategy.id, chunks=chunks))
    return DocumentResult(path=path, metadata=metadata, strategy_results=strategy_results)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_single_report(result: DocumentResult) -> str:
    metadata = result.metadata
    lines: list[str] = [
        f"# Chunk Comparison: {metadata.source_file}",
        "",
        f"**Document type:** {metadata.document_type}  ",
        f"**Country:** {metadata.country or 'N/A'} | **City:** {metadata.city or 'N/A'}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Strategy         | Chunks | Avg (chars) | Min | Max |",
        "|------------------|--------|-------------|-----|-----|",
    ]
    for r in result.strategy_results:
        lines.append(f"| {r.strategy_id:<16} | {r.count:<6} | {r.avg_size:<11} | {r.min_size:<3} | {r.max_size:<3} |")

    lines += ["", "---", ""]

    for r in result.strategy_results:
        lines += [f"## {r.strategy_id}", ""]
        for i, (text, meta) in enumerate(r.chunks, start=1):
            lines += [
                f"### Chunk {i}",
                "",
                f"**Section:** {meta.section or 'N/A'}  ",
                f"**Size:** {len(text)} chars  ",
                "",
                "**Content:**",
                "```",
                text.strip(),
                "```",
                "",
            ]
        lines += ["---", ""]

    return "\n".join(lines)


def render_batch_report(doc_type: str, results: list[DocumentResult]) -> str:
    strategies = [r.strategy_id for r in results[0].strategy_results]

    # Header columns: one per strategy
    header_cells = " | ".join(f"{'chunks / avg':<16}" for _ in strategies)
    strategy_headers = " | ".join(f"{s:<16}" for s in strategies)
    separator = "|-------------------------------" + ("|------------------" * len(strategies)) + "|"

    lines: list[str] = [
        f"# Batch Comparison: {doc_type}",
        "",
        f"**Files processed:** {len(results)}  ",
        f"**Strategies:** {', '.join(strategies)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| {'File':<30} | {strategy_headers} |",
        f"|{'-'*31}|" + "|".join([f"{'-'*18}" for _ in strategies]) + "|",
    ]

    for doc_result in results:
        stem = doc_result.path.stem
        cells = []
        for r in doc_result.strategy_results:
            cells.append(f"{r.count} / {r.avg_size:<12}")
        lines.append(f"| {stem:<30} | {' | '.join(cells)} |")

    lines += ["", "---", ""]

    # Per-file min/max breakdown
    lines += ["## Per-file Detail", ""]
    for doc_result in results:
        lines += [f"### {doc_result.path.stem}", ""]
        lines += [
            "| Strategy         | Chunks | Avg (chars) | Min | Max |",
            "|------------------|--------|-------------|-----|-----|",
        ]
        for r in doc_result.strategy_results:
            lines.append(
                f"| {r.strategy_id:<16} | {r.count:<6} | {r.avg_size:<11} | {r.min_size:<3} | {r.max_size:<3} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_single(path: Path, output_dir: Path) -> None:
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(1)

    strategies = build_strategies()
    print(f"Processing {path.name}...")
    result = process_file(path, strategies)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"compare_{path.stem}.md"
    out.write_text(render_single_report(result), encoding="utf-8")
    print(f"Report written to {out}")


def run_batch(doc_type: str, output_dir: Path) -> None:
    folder = _TYPE_TO_FOLDER.get(doc_type)
    if not folder:
        print(f"Unknown document type: {doc_type}. Valid: {list(_TYPE_TO_FOLDER)}", file=sys.stderr)
        sys.exit(1)

    data_dir = ROOT / "data" / folder
    paths = sorted(data_dir.glob("*.md"))
    if not paths:
        print(f"No .md files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Batch mode: {doc_type} — {len(paths)} files found.")
    print("Warning: semantic chunking calls OpenAI once per file.\n")

    strategies = build_strategies()  # single instantiation, shared across files
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[DocumentResult] = []
    for path in paths:
        print(f"  {path.name}")
        result = process_file(path, strategies)
        results.append(result)

        # Write individual report alongside batch
        individual_out = output_dir / f"compare_{path.stem}.md"
        individual_out.write_text(render_single_report(result), encoding="utf-8")
        print(f"    → {individual_out.name}")

    batch_out = output_dir / f"batch_{doc_type}.md"
    batch_out.write_text(render_batch_report(doc_type, results), encoding="utf-8")
    print(f"\nBatch report written to {batch_out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on a markdown file or a full document type batch."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Path to a single .md file",
    )
    group.add_argument(
        "--type",
        dest="doc_type",
        choices=list(_TYPE_TO_FOLDER),
        help="Process all files of a given document type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write reports (default: reports/)",
    )
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()

    if args.doc_type:
        run_batch(args.doc_type, output_dir)
    else:
        run_single(args.path.expanduser().resolve(), output_dir)


if __name__ == "__main__":
    main()