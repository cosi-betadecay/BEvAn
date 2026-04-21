#!/usr/bin/env python3
"""Generate a Markdown test summary from a pytest JUnit XML report."""

from __future__ import annotations

import argparse
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path


def _as_int(value: str | None) -> int:
    return int(value) if value not in (None, "") else 0


def _as_float(value: str | None) -> float:
    return float(value) if value not in (None, "") else 0.0


def _collect_testsuites(root: ET.Element) -> list[ET.Element]:
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        return list(root.findall("testsuite"))
    raise ValueError(f"Unsupported JUnit root tag: {root.tag}")


def _collect_failures(testsuites: list[ET.Element]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    for suite in testsuites:
        for case in suite.findall("testcase"):
            classname = case.get("classname", "").strip()
            name = case.get("name", "").strip()

            for tag in ("failure", "error"):
                node = case.find(tag)
                if node is None:
                    continue

                message = (node.get("message") or node.text or "").strip()
                message = " ".join(message.split())
                failures.append(
                    {
                        "test": f"{classname}::{name}" if classname else name,
                        "kind": tag,
                        "message": message,
                    }
                )
                break

    return failures


def build_summary(xml_path: Path, environment: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    testsuites = _collect_testsuites(root)

    total = sum(_as_int(suite.get("tests")) for suite in testsuites)
    failures_count = sum(_as_int(suite.get("failures")) for suite in testsuites)
    errors_count = sum(_as_int(suite.get("errors")) for suite in testsuites)
    skipped_count = sum(_as_int(suite.get("skipped")) for suite in testsuites)
    duration = sum(_as_float(suite.get("time")) for suite in testsuites)
    passed = total - failures_count - errors_count - skipped_count

    timestamp = root.get("timestamp")
    if not timestamp and testsuites:
        timestamp = testsuites[0].get("timestamp")

    failures = _collect_failures(testsuites)

    lines = [
        "# Test Summary",
        "",
        "<!-- This file is generated from a pytest JUnit XML report. -->",
        "",
        f"- Source report: `{xml_path.as_posix()}`",
        f"- Environment: {environment}",
        f"- Last run: {timestamp or 'unknown'}",
        "",
        "## Overall",
        "",
        f"- Total: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failures_count}",
        f"- Errors: {errors_count}",
        f"- Skipped: {skipped_count}",
        f"- Duration: {duration:.3f}s",
        "",
    ]

    if testsuites:
        lines.extend(["## Test Suites", ""])
        for suite in testsuites:
            suite_name = suite.get("name", "unnamed suite")
            suite_total = _as_int(suite.get("tests"))
            suite_failures = _as_int(suite.get("failures"))
            suite_errors = _as_int(suite.get("errors"))
            suite_skipped = _as_int(suite.get("skipped"))
            suite_passed = suite_total - suite_failures - suite_errors - suite_skipped
            suite_duration = _as_float(suite.get("time"))
            lines.append(
                f"- `{suite_name}`: {suite_passed} passed, {suite_failures} failed, "
                f"{suite_errors} errors, {suite_skipped} skipped in {suite_duration:.3f}s"
            )
        lines.append("")

    if failures:
        lines.extend(["## Failures And Errors", ""])
        for failure in failures:
            lines.append(f"- `{failure['test']}` ({failure['kind']})")
            if failure["message"]:
                lines.append(f"  {textwrap.shorten(failure['message'], width=220, placeholder='...')}")
        lines.append("")
    else:
        lines.extend(["## Failures And Errors", "", "- None.", ""])

    lines.extend(
        [
            "## Notes",
            "",
            "- These results come from a local ROOT-enabled environment.",
            "- Regenerate this file after running pytest to keep the documentation current.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path", type=Path, help="Path to the pytest JUnit XML report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/test-summary.md"),
        help="Where to write the Markdown summary.",
    )
    parser.add_argument(
        "--environment",
        default="local ROOT-enabled environment",
        help="Short environment label to include in the summary.",
    )
    args = parser.parse_args()

    summary = build_summary(args.xml_path, environment=args.environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
