#!/usr/bin/env python3
"""Validate Milestone Review Reports against structural quality rules."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ───────────────────────────────────────────────────────────────────────
# Section A: Parsing Utilities
# ───────────────────────────────────────────────────────────────────────

VALID_VERDICTS = {"delivering", "at risk", "broken"}
VALID_ADJUSTMENT_LEVELS = {"stay", "adjust", "reconsider"}

MAX_FOCUS_TASKS = 3


def is_non_empty_value(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if normalized in {"-", "N/A", "n/a", "TBD", "tbd", "TODO", "todo"}:
        return False
    if re.fullmatch(r"\[[^\]]+\]", normalized):
        return False
    return True


def section_block_fuzzy(text: str, keyword: str) -> str:
    pattern = re.compile(
        rf"(?ms)^##\s*[^#\n]*{re.escape(keyword)}[^#\n]*$\n(.*?)(?=^##\s|\Z)"
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


def find_field_value(block: str, label: str) -> Optional[str]:
    pattern = re.compile(
        rf"(?m)^(?P<indent>\s*)-\s*{re.escape(label)}[^\S\n]*(?P<rest>.*)$"
    )
    match = pattern.search(block)
    if not match:
        return None

    first = match.group("rest").strip()
    if first:
        return first

    trailing = block[match.end():].splitlines()
    collected: List[str] = []
    for line in trailing:
        if re.match(r"^\s*-\s*[^:]+:\s*", line):
            break
        normalized = line.strip()
        if normalized.startswith("-"):
            normalized = normalized[1:].strip()
        if normalized:
            collected.append(normalized)
    if not collected:
        return ""
    return " | ".join(collected)


def count_list_items(block: str) -> int:
    items = re.findall(r"(?m)^(?:\d+\.|[-*+])[^\S\n]+(.+)$", block)
    return sum(1 for item in items if is_non_empty_value(item))


# ───────────────────────────────────────────────────────────────────────
# Section B: Validation Checks
# ───────────────────────────────────────────────────────────────────────


def _check_promise_verdict(text: str) -> List[Tuple[str, str]]:
    """Check #1: Promise Delivery verdict must be a valid enum value."""
    issues = []
    block = section_block_fuzzy(text, "Promise Delivery")
    if not block:
        return issues

    verdict_value = find_field_value(block, "판정:")
    if verdict_value is None:
        verdict_value = find_field_value(block, "Verdict:")

    if verdict_value and is_non_empty_value(verdict_value):
        if verdict_value.strip().lower() not in VALID_VERDICTS:
            issues.append((
                "error",
                f"Promise Delivery verdict '{verdict_value.strip()}' is not valid; "
                f"must be one of: Delivering / At Risk / Broken",
            ))
    return issues


def _check_adjustment_level(text: str) -> List[Tuple[str, str]]:
    """Check #2: Direction adjustment level must be a valid enum value."""
    issues = []
    block = section_block_fuzzy(text, "방향 조정")
    if not block:
        block = section_block_fuzzy(text, "Direction Adjustment")
    if not block:
        return issues

    level = find_field_value(block, "수준:")
    if level is None:
        level = find_field_value(block, "Level:")

    if level and is_non_empty_value(level):
        if level.strip().lower() not in VALID_ADJUSTMENT_LEVELS:
            issues.append((
                "error",
                f"Direction adjustment level '{level.strip()}' is not valid; "
                f"must be one of: Stay / Adjust / Reconsider",
            ))
    return issues


def _check_focus_tasks_count(text: str) -> List[Tuple[str, str]]:
    """Check #3: Focus tasks should not exceed 3."""
    issues = []
    block = section_block_fuzzy(text, "집중 과제")
    if not block:
        block = section_block_fuzzy(text, "Focus")
    if not block:
        return issues

    count = count_list_items(block)
    if count > MAX_FOCUS_TASKS:
        issues.append((
            "warning",
            f"Too many focus tasks ({count}); recommend {MAX_FOCUS_TASKS} or fewer to maintain focus",
        ))
    return issues


# ───────────────────────────────────────────────────────────────────────
# Section C: Validation Entry Point
# ───────────────────────────────────────────────────────────────────────


def validate_milestone_review(text: str) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    checks = [
        _check_promise_verdict(text),
        _check_adjustment_level(text),
        _check_focus_tasks_count(text),
    ]

    for result in checks:
        for severity, message in result:
            if severity == "error":
                errors.append(message)
            else:
                warnings.append(message)

    return errors, warnings


# ───────────────────────────────────────────────────────────────────────
# Section D: Output Formatting
# ───────────────────────────────────────────────────────────────────────


def format_text(errors: List[str], warnings: List[str]) -> str:
    status = "PASS" if not errors else "FAIL"
    lines: List[str] = []
    lines.append(f"VALIDATION: {status}")
    lines.append(f"ERRORS: {len(errors)}")
    lines.append(f"WARNINGS: {len(warnings)}")
    if errors:
        lines.append("")
        lines.append("Blocking Issues:")
        for item in errors:
            lines.append(f"- {item}")
    if warnings:
        lines.append("")
        lines.append("Recommended Improvements:")
        for item in warnings:
            lines.append(f"- {item}")
    return "\n".join(lines)


def format_json(errors: List[str], warnings: List[str]) -> str:
    status = "PASS" if not errors else "FAIL"
    return json.dumps(
        {"status": status, "errors": errors, "warnings": warnings},
        ensure_ascii=False,
        indent=2,
    )


# ───────────────────────────────────────────────────────────────────────
# Section E: CLI
# ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Milestone Review Report markdown file",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        text = input_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: failed to read input: {exc}", file=sys.stderr)
        return 2

    errors, warnings = validate_milestone_review(text)

    if args.format == "json":
        print(format_json(errors, warnings))
    else:
        print(format_text(errors, warnings))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
