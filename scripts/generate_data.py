#!/usr/bin/env python3
"""
Generate synthetic NomadLATAM markdown documents via Claude (Anthropic API).

Requires: pip install -e . (dependencies from pyproject.toml)
Environment: ANTHROPIC_API_KEY (optional .env at project root via load_dotenv)

Output: data/{city_guides,visa_info,coworking,cost_comparison}/*.md
Idempotent: fixed filenames; re-running overwrites the same files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from anthropic import Anthropic
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependency: pip install -e .", file=sys.stderr)
    sys.exit(1)

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"

CITY_GUIDES = DATA / "city_guides"
VISA_INFO = DATA / "visa_info"
COWORKING = DATA / "coworking"
COST_COMPARISON = DATA / "cost_comparison"

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM = (
    "You produce realistic synthetic markdown for RAG pipeline testing. "
    "Follow the user's formatting instructions exactly. "
    "Accuracy is not critical; plausible structure and entity-rich text matter. "
    "Output only the document body in markdown — no title line outside the doc, no preamble."
)


def _client() -> Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("Set ANTHROPIC_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)
    return Anthropic(api_key=key)


def generate(client: Anthropic, user_prompt: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    block = msg.content[0]
    if block.type != "text":
        raise RuntimeError(f"Unexpected response block type: {block.type}")
    return block.text.strip()


# --- One prompt template per document type (parameterized) ---


def prompt_city_guide(city: str, country: str, style: str) -> str:
    """Long-form city guide; style hints structural variety."""
    return f"""Write a digital-nomad-oriented city guide for {city}, {country}.

Target length: roughly 900–1400 words.

Structure instructions ({style}):
- Include sections covering: Cost of Living, Internet & Coworking, Neighborhoods, Visa Info, Food & Culture, Safety, Climate.
- Use markdown. {style}

Make up plausible venue names, neighborhoods, and price ranges. Do not add a YAML front matter."""


def prompt_visa_info_table(region_focus: str) -> str:
    return f"""Write a visa reference document focused on: {region_focus}.

Format: a proper markdown table with columns: Country | Visa type | Typical duration | Key requirements | Approx. cost | Processing notes.
Include at least 5 rows covering several Latin American countries (mix of digital nomad, tourist extension, temporary residence where plausible).
Add a short introductory paragraph (2–3 sentences) before the table. Use ## headings for any section that introduces or follows the table."""

def prompt_visa_info_bullets(country: str) -> str:
    return f"""Write a visa options overview for {country} aimed at remote workers and tourists.

Format: NO tables. Use ## headings for each visa pathway section. Use bullet lists and short paragraphs within each section.
Cover common paths: tourist stay, extensions, temporary residence — all generic and non-legal-advice tone.
Length: ~400–700 words."""


def prompt_coworking_reviews(cities: list[str], compact: bool) -> str:
    cities_str = ", ".join(cities)
    length = "Keep each review under ~120 words." if compact else "Allow 150–220 words per review."
    return f"""Write a single markdown document with coworking space reviews for these cities: {cities_str}.

Include 4 distinct coworking spaces (mix across cities). For each, use a ### heading with a fictional space name.
Fields woven into prose or bullets: approximate monthly hot-desk price, internet quality, amenities, vibe.
{length}
Entity-dense; realistic but invented names."""


def prompt_cost_comparison(title: str, cities: list[str], prose_style: str) -> str:
    cities_str = ", ".join(cities)
    return f"""Write a cost-of-living comparison titled conceptually: "{title}".

Cities to compare: {cities_str}.

Cover categories: Rent (1BR central-ish), Groceries, Transport, Dining out, Coworking, Home internet.
Use {prose_style}. Include approximate monthly USD ranges as rough ballparks (clearly informal).
Length: ~500–900 words. Markdown with ## section headers for categories."""


# --- Document plan: 12 files, mixed formatting ---


def ensure_dirs() -> None:
    for d in (CITY_GUIDES, VISA_INFO, COWORKING, COST_COMPARISON):
        d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    client = _client()

    jobs: list[tuple[Path, str]] = [
        # City guides — clean headers vs slightly looser
        (
            CITY_GUIDES / "medellin_colombia.md",
            prompt_city_guide(
                "Medellín",
                "Colombia",
                "Use consistent ## headers for each major section.",
            ),
        ),
        (
            CITY_GUIDES / "florianopolis_brazil.md",
            prompt_city_guide(
                "Florianópolis",
                "Brazil",
                "Use ## for most sections but allow one section to use bold pseudo-headers instead of ## for variety.",
            ),
        ),
        (
            CITY_GUIDES / "mexico_city_mexico.md",
            prompt_city_guide(
                "Mexico City",
                "Mexico",
                "Strict ## headers and a small ### subsection under Neighborhoods.",
            ),
        ),
        (
            CITY_GUIDES / "buenos_aires_argentina.md",
            prompt_city_guide(
                "Buenos Aires",
                "Argentina",
                "Slightly looser: mix ## headers with occasional thematic paragraphs without headers.",
            ),
        ),
        # Visa — table vs bullets vs mixed
        (
            VISA_INFO / "latin_america_dnv_summary_table.md",
            prompt_visa_info_table("digital nomad and medium-stay friendly visas in Latin America"),
        ),
        (
            VISA_INFO / "argentina_visa_options_bullets.md",
            prompt_visa_info_bullets("Argentina"),
        ),
        (
            VISA_INFO / "mexico_visa_paths_overview.md",
            prompt_visa_info_bullets("Mexico"),
        ),
        # Coworking — compact vs longer reviews
        (
            COWORKING / "medellin_florianopolis_reviews.md",
            prompt_coworking_reviews(["Medellín", "Florianópolis"], compact=True),
        ),
        (
            COWORKING / "mexico_city_buenos_aires_reviews.md",
            prompt_coworking_reviews(["Mexico City", "Buenos Aires"], compact=False),
        ),
        # Cost comparison
        (
            COST_COMPARISON / "medellin_vs_mexico_city.md",
            prompt_cost_comparison(
                "Medellín vs Mexico City monthly budget",
                ["Medellín", "Mexico City"],
                "narrative paragraphs with bullet lists under each ## category",
            ),
        ),
        (
            COST_COMPARISON / "four_city_rent_and_coworking_snapshot.md",
            prompt_cost_comparison(
                "Four-city snapshot",
                ["Medellín", "Florianópolis", "Mexico City", "Buenos Aires"],
                "more tabular feel: use markdown tables for rent and coworking only; prose for other categories",
            ),
        ),
    ]

    for path, prompt in jobs:
        rel = path.relative_to(ROOT)
        print(f"Generating {rel} ...", flush=True)
        text = generate(client, prompt)
        path.write_text(text + "\n", encoding="utf-8")

    print(f"Wrote {len(jobs)} documents under {DATA}", flush=True)


if __name__ == "__main__":
    main()
