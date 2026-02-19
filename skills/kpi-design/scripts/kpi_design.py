#!/usr/bin/env python3
"""Validate game KPI plans against quality rules."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# ───────────────────────────────────────────────────────────────────────
# Section A: Reference Data (validate-only)
# Source of truth: kpi-catalog.md § Stage Profiles
# ───────────────────────────────────────────────────────────────────────

STAGE_EXCLUDED_CATEGORIES = {
    "prototype": {"monetization"},
    "soft-launch": set(),
    "live": set(),
}

STAGE_REQUIRED_CATEGORIES = {
    "prototype": {"activation", "engagement"},
    "soft-launch": {"retention", "engagement"},
    "live": {"retention", "monetization", "engagement"},
}

STAGE_MAX_KPIS = {
    "prototype": 5,
    "soft-launch": 6,
    "live": 7,
}

CATEGORY_KEYWORDS = {
    "retention": ["retention", "d1", "d7", "d30", "churn", "return rate", "returning"],
    "activation": [
        "tutorial", "first session", "onboarding", "core loop completion",
        "activation", "first_session", "core_loop",
    ],
    "engagement": [
        "session", "run", "level", "match", "replay", "duration",
        "engagement", "combo", "diversity", "death", "attempt",
    ],
    "monetization": [
        "arpu", "arppu", "ltv", "revenue", "iap", "purchase", "monetization",
        "ad view", "ad_view", "ad-to-session", "ad_impression", "store", "payer",
    ],
}

# ───────────────────────────────────────────────────────────────────────
# Section B: Parsing Utilities
# ───────────────────────────────────────────────────────────────────────


def is_non_empty_value(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if normalized in {"-", "N/A", "n/a", "TBD", "tbd", "TODO", "todo"}:
        return False
    if re.fullmatch(r"\[[^\]]+\]", normalized):
        return False
    return True


def section_block(text: str, heading: str) -> str:
    # Allow numbered markdown headings like "## 10) KPI Plan" used by the agent template.
    pattern = re.compile(
        rf"(?ms)^##\s*(?:\d+\s*[\)\.\-:]?\s*)?{re.escape(heading)}\s*$\n(.*?)(?=^##\s|\Z)"
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


def find_field_value(block: str, label: str) -> Optional[str]:
    pattern = re.compile(
        rf"(?m)^(?P<indent>\s*)-\s*{re.escape(label)}\s*(?P<rest>.*)$"
    )
    match = pattern.search(block)
    if not match:
        return None

    first = match.group("rest").strip()
    if first:
        return first

    trailing = block[match.end() :].splitlines()
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


def infer_kpi_categories(kpi: dict) -> Set[str]:
    text = f"{kpi.get('name', '')} {kpi.get('formula', '')}".lower()
    categories = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            categories.add(category)
    return categories


# ───────────────────────────────────────────────────────────────────────
# Section C: Validate Mode
# ───────────────────────────────────────────────────────────────────────


def _check_decision_rules(kpi_blocks: List[dict]) -> List[Tuple[str, str]]:
    """Check #1: Decision rule missing → Error."""
    issues = []
    for kpi in kpi_blocks:
        dr = kpi.get("decision_rule", "")
        if not dr or not is_non_empty_value(dr):
            issues.append(("error", f"KPI '{kpi['name']}' is missing a Decision Rule"))
    return issues


def _check_vanity_metrics(kpi_blocks: List[dict]) -> List[Tuple[str, str]]:
    """Check #2: Vanity metric detection → Warning."""
    issues = []
    vanity_patterns = [
        r"^total[_ ]",
        r"^count[_ ]",
        r"^number[_ ]of",
        r"^cumulative[_ ]",
    ]
    for kpi in kpi_blocks:
        formula = kpi.get("formula", "").lower()
        name = kpi.get("name", "").lower()
        dr = kpi.get("decision_rule", "")
        is_vanity = any(re.search(p, formula) or re.search(p, name) for p in vanity_patterns)
        no_decision = not dr or not is_non_empty_value(dr)
        if is_vanity and no_decision:
            issues.append((
                "warning",
                f"KPI '{kpi['name']}' looks like a vanity metric (absolute count without decision linkage)",
            ))
    return issues


def _check_kpi_count(kpi_blocks: List[dict], stage: Optional[str] = None) -> List[Tuple[str, str]]:
    """Check #3: KPI count exceeds stage limit → Error."""
    issues = []
    count = len(kpi_blocks)
    max_kpis = STAGE_MAX_KPIS.get(stage, 7) if stage else 7
    if count > max_kpis:
        issues.append(("error", f"Too many KPIs ({count}); maximum for '{stage or 'default'}' is {max_kpis}"))
    return issues


def _check_retention_metric(kpi_blocks: List[dict]) -> List[Tuple[str, str]]:
    """Check #4: No retention metric → Error."""
    issues = []
    has_retention = any("retention" in infer_kpi_categories(kpi) for kpi in kpi_blocks)
    if not has_retention:
        issues.append(("error", "No retention metric found; include at least 1 retention KPI"))
    return issues


def _check_early_session_metric(kpi_blocks: List[dict]) -> List[Tuple[str, str]]:
    """Check #5: No early-session quality metric → Error."""
    issues = []
    has_early = any("activation" in infer_kpi_categories(kpi) for kpi in kpi_blocks)
    if not has_early:
        issues.append((
            "error",
            "No early-session quality metric found; include at least 1 activation/onboarding KPI",
        ))
    return issues


def _check_instrumentation_events(
    text: str, kpi_blocks: List[dict],
) -> List[Tuple[str, str]]:
    """Check #7: Instrumentation Events mapping → Warning."""
    issues = []
    events_block = section_block(text, "Instrumentation Events")
    if not events_block.strip():
        issues.append((
            "warning",
            "Instrumentation Events section is missing or empty; event mapping required for KPI measurement",
        ))
        return issues

    # Parse "- event_name: KPI1, KPI2" entries and collect referenced KPI names
    referenced_kpis: Set[str] = set()
    for match in re.finditer(r"(?m)^-\s*[^:]+:\s*(.+)$", events_block):
        kpi_list = match.group(1)
        for kpi_ref in kpi_list.split(","):
            kpi_ref = kpi_ref.strip()
            if kpi_ref:
                referenced_kpis.add(kpi_ref.lower())

    # Check that each KPI in kpi_blocks is referenced by at least one event
    for kpi in kpi_blocks:
        kpi_name = kpi.get("name", "").strip()
        if kpi_name and kpi_name.lower() not in referenced_kpis:
            issues.append((
                "warning",
                f"KPI '{kpi_name}' is not referenced in Instrumentation Events",
            ))

    return issues


def _check_stage_appropriateness(
    kpi_blocks: List[dict], stage: Optional[str],
) -> List[Tuple[str, str]]:
    """Check #6: Stage profile fit (required categories + excluded categories)."""
    issues = []
    if not stage:
        issues.append((
            "info",
            "Stage not specified; skipping stage-specific checks. "
            "Use --stage or add 'Stage:' field to enable.",
        ))
        return issues

    required = STAGE_REQUIRED_CATEGORIES.get(stage)
    excluded = STAGE_EXCLUDED_CATEGORIES.get(stage)
    if required is None or excluded is None:
        issues.append((
            "warning",
            f"Unrecognized stage '{stage}'; "
            f"valid stages are: {', '.join(VALID_STAGES)}. "
            f"Skipping stage-specific checks.",
        ))
        return issues

    present_categories = set()
    for kpi in kpi_blocks:
        present_categories.update(infer_kpi_categories(kpi))

    missing_required = sorted(required - present_categories)
    for category in missing_required:
        issues.append((
            "error",
            f"Stage '{stage}' requires at least 1 '{category}' KPI",
        ))

    for kpi in kpi_blocks:
        categories = infer_kpi_categories(kpi)
        for category in sorted(excluded.intersection(categories)):
            issues.append((
                "warning",
                f"KPI '{kpi['name']}' is a {category} metric, "
                f"which is typically excluded at the '{stage}' stage",
            ))
    return issues


def parse_kpi_markdown(text: str) -> Tuple[List[dict], Optional[str]]:
    """Parse KPI plan markdown into structured blocks."""
    kpi_blocks: List[dict] = []

    # Extract stage from the KPI Plan section
    plan_block = section_block(text, "KPI Plan")
    stage_value = find_field_value(plan_block, "Stage:") if plan_block else None
    stage = stage_value.strip().lower() if stage_value and is_non_empty_value(stage_value) else None

    # Parse primary metric from KPI Plan section
    if plan_block:
        primary_name = find_field_value(plan_block, "Primary Outcome Metric:")
        if primary_name and is_non_empty_value(primary_name):
            primary_kpi = {
                "name": primary_name.strip(),
                "formula": "",
                "decision_rule": "",
            }
            primary_def = find_field_value(plan_block, "Definition:")
            primary_dr = find_field_value(plan_block, "Decision Rule:")
            if primary_def:
                primary_kpi["formula"] = primary_def.strip()
            if primary_dr:
                primary_kpi["decision_rule"] = primary_dr.strip()
            kpi_blocks.append(primary_kpi)

    # Parse supporting KPIs
    supporting_block = section_block(text, "Supporting KPIs")
    if supporting_block:
        # Split by top-level list items (KPI names)
        kpi_chunks = re.split(r"(?m)^- (?=[^\s])", supporting_block)
        for chunk in kpi_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            # Guard: skip chunks without KPI sub-fields (likely intro text)
            if "Formula:" not in chunk and "Decision Rule:" not in chunk:
                continue
            lines = chunk.splitlines()
            name = lines[0].strip()
            if not name or not is_non_empty_value(name):
                continue
            kpi: dict = {"name": name, "formula": "", "decision_rule": ""}
            full_chunk = "- " + chunk
            formula = find_field_value(full_chunk, "Formula:")
            dr = find_field_value(full_chunk, "Decision Rule:")
            if formula:
                kpi["formula"] = formula.strip()
            if dr:
                kpi["decision_rule"] = dr.strip()
            kpi_blocks.append(kpi)

    return kpi_blocks, stage


def validate_kpi_plan(text: str, stage: Optional[str] = None) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    kpi_blocks, parsed_stage = parse_kpi_markdown(text)
    effective_stage = stage or parsed_stage

    if not kpi_blocks:
        errors.append("No KPIs found in the document")
        return errors, warnings

    checks = [
        _check_decision_rules(kpi_blocks),
        _check_vanity_metrics(kpi_blocks),
        _check_kpi_count(kpi_blocks, effective_stage),
        _check_retention_metric(kpi_blocks),
        _check_early_session_metric(kpi_blocks),
        _check_stage_appropriateness(kpi_blocks, effective_stage),
        _check_instrumentation_events(text, kpi_blocks),
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


def format_validate_text(errors: List[str], warnings: List[str]) -> str:
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


def format_validate_json(errors: List[str], warnings: List[str]) -> str:
    status = "PASS" if not errors else "FAIL"
    return json.dumps(
        {"status": status, "errors": errors, "warnings": warnings},
        ensure_ascii=False,
        indent=2,
    )


# ───────────────────────────────────────────────────────────────────────
# Section E: CLI
# ───────────────────────────────────────────────────────────────────────

VALID_STAGES = list(STAGE_EXCLUDED_CATEGORIES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to KPI plan markdown file",
    )
    parser.add_argument(
        "--stage",
        choices=VALID_STAGES,
        help="Game development stage (optional; auto-detected from document if omitted)",
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

    errors, warnings = validate_kpi_plan(text, stage=args.stage)

    if args.format == "json":
        print(format_validate_json(errors, warnings))
    else:
        print(format_validate_text(errors, warnings))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
