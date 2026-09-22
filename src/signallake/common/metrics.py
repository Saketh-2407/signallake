"""Shared helper: replace-or-append a `## <section_key>: ...` section in reports/METRICS.md.

Used by every phase that measures something real (Phase 5's load test, Phase 6's recompute
benchmark, ...) so re-running a measurement updates its own section in place instead of
duplicating it.
"""

import re
from pathlib import Path

HEADER = "# SignalLake -- Measured Metrics\n\n"


def upsert_metrics_section(
    metrics_path: Path, section_key: str, section_title: str, body: str
) -> None:
    """Replace `## {section_key}: ...` with `## {section_title}` + body, or append it."""
    section = f"## {section_title}\n\n{body.strip()}\n"
    existing = metrics_path.read_text() if metrics_path.exists() else HEADER
    pattern = re.compile(rf"## {re.escape(section_key)}:.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(existing):
        new_content = pattern.sub(section, existing)
    else:
        new_content = existing.rstrip() + "\n\n" + section
    metrics_path.write_text(new_content)
