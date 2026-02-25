"""CLI for distill content quality scoring."""

from __future__ import annotations

import io
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import scorers to trigger registration
import distill.scorers  # noqa: F401
from distill.pipeline import Pipeline

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()


def _score_bar(score: float, width: int = 20) -> Text:
    """Render a visual score bar."""
    filled = int(score * width)
    empty = width - filled

    if score >= 0.7:
        color = "green"
    elif score >= 0.5:
        color = "yellow"
    else:
        color = "red"

    bar = Text()
    bar.append("#" * filled, style=color)
    bar.append("." * empty, style="dim")
    bar.append(f" {score:.2f}", style="bold")
    return bar


def _display_report(report, source: str = ""):
    """Rich display of a quality report."""
    # Header
    grade_colors = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red bold"}
    grade_style = grade_colors.get(report.grade, "white")

    title = "Quality Report"
    if source:
        title += f" - {source[:60]}"

    # Overall score panel
    overall = Text()
    overall.append("\n  Grade: ", style="bold")
    overall.append(f"{report.grade}", style=f"bold {grade_style}")
    overall.append(f"  ({report.label})\n", style="dim")
    overall.append("  Overall: ")
    overall.append(_score_bar(report.overall_score))
    overall.append(f"\n  Words: {report.word_count:,}\n")

    console.print(Panel(overall, title=title, border_style="blue"))

    # Dimension breakdown
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", min_width=25)
    table.add_column("Details", style="dim")

    for result in report.scores:
        table.add_row(
            result.name,
            _score_bar(result.score),
            result.explanation[:80],
        )

    console.print(table)
    console.print()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """distill - Content quality scoring toolkit.

    Separates signal from noise by measuring substance,
    epistemic honesty, and readability.
    """
    pass


@main.command()
@click.argument("source")
@click.option("--scorers", "-s", help="Comma-separated scorer names", default=None)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def score(source: str, scorers: str | None, as_json: bool):
    """Score content quality from a URL or file.

    SOURCE can be a URL (https://...) or a file path (- for stdin).
    """
    scorer_names = scorers.split(",") if scorers else None

    # Get text
    if source == "-":
        text = sys.stdin.read()
        label = "stdin"
    elif source.startswith(("http://", "https://")):
        from distill.extractors import extract_from_url

        console.print(f"[dim]Fetching {source}...[/dim]")
        try:
            extracted = extract_from_url(source)
            text = extracted["text"]
            label = extracted.get("title", source)
        except Exception as e:
            console.print(f"[red]Error fetching URL: {e}[/red]")
            raise SystemExit(1)
    else:
        try:
            with open(source) as f:
                text = f.read()
            label = source
        except FileNotFoundError:
            console.print(f"[red]File not found: {source}[/red]")
            raise SystemExit(1)

    # Score
    pipeline = Pipeline(scorers=scorer_names)
    report = pipeline.score(text)

    if as_json:
        import json

        data = {
            "overall_score": round(report.overall_score, 3),
            "grade": report.grade,
            "label": report.label,
            "word_count": report.word_count,
            "dimensions": {
                r.name: {
                    "score": round(r.score, 3),
                    "explanation": r.explanation,
                    "details": r.details,
                }
                for r in report.scores
            },
        }
        click.echo(json.dumps(data, indent=2))
    else:
        _display_report(report, source=label)


@main.command(name="list")
def list_scorers():
    """List available scorers."""
    from distill.scorer import list_scorers as _list

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for name, desc in _list().items():
        table.add_row(name, desc)

    console.print(table)


@main.command()
def demo():
    """Run a demo comparing high-quality vs low-quality content."""
    console.print("\n[bold]distill demo[/bold] - comparing content quality\n")

    ai_slop = """
    In today's rapidly evolving digital landscape, it's important to understand the
    key factors that drive success in software development. Whether you're a seasoned
    professional or just starting out, there are several best practices you should
    keep in mind. First and foremost, code quality is essential. This means writing
    clean, maintainable code that follows established patterns. Another key factor is
    collaboration. Working effectively with your team can take your projects to the
    next level. In conclusion, by following these proven strategies, you can unlock
    your full potential as a developer.
    """

    expert_content = """
    We migrated our payment service from a monolith to a separate deployment in Q3 2024.
    Latency dropped from p99 of 340ms to 95ms, but we hit an unexpected issue: our
    connection pool was sized for the monolith's traffic patterns (200 concurrent
    connections shared across 15 services), and the isolated service only needed 30.
    The oversized pool was actually masking a connection leak in our retry logic.

    The tradeoff worth noting: our deployment complexity increased significantly.
    We went from one CI pipeline to three, and debugging cross-service issues now
    requires correlating logs across systems. For teams smaller than ours (we have
    6 backend engineers), I'd honestly recommend staying with the monolith until
    the pain is concrete and measurable, not theoretical.
    """

    pipeline = Pipeline()

    console.print("[bold red]Sample A: Generic AI-generated content[/bold red]")
    report_a = pipeline.score(ai_slop)
    _display_report(report_a, "Generic AI content")

    console.print("[bold green]Sample B: Expert practitioner content[/bold green]")
    report_b = pipeline.score(expert_content)
    _display_report(report_b, "Expert practitioner content")


if __name__ == "__main__":
    main()
