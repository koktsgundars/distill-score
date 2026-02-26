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


def _resolve_source(source: str, quiet: bool = False) -> tuple[str, str, dict | None]:
    """Resolve a source (URL, file path, or '-' for stdin) to (label, text, metadata).

    Returns:
        Tuple of (label, text content, metadata dict or None).

    Raises:
        SystemExit: If the source cannot be resolved.
    """
    if source == "-":
        return "stdin", sys.stdin.read(), None
    elif source.startswith(("http://", "https://")):
        from distill.extractors import extract_from_url

        if not quiet:
            console.print(f"[dim]Fetching {source}...[/dim]")
        try:
            extracted = extract_from_url(source)
            metadata = {"url": extracted.get("url", source), "title": extracted.get("title", "")}
            return extracted.get("title", source), extracted["text"], metadata
        except Exception as e:
            console.print(f"[red]Error fetching URL: {e}[/red]")
            raise SystemExit(1)
    else:
        try:
            with open(source) as f:
                return source, f.read(), None
        except FileNotFoundError:
            console.print(f"[red]File not found: {source}[/red]")
            raise SystemExit(1)


def _report_to_dict(report, source: str | None = None) -> dict:
    """Convert a QualityReport to a JSON-serializable dict."""
    data: dict = {}
    if source is not None:
        data["source"] = source
    data.update({
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
    })
    if report.paragraph_scores:
        data["paragraphs"] = [
            {
                "index": ps.index,
                "preview": ps.text_preview,
                "overall_score": round(ps.overall_score, 3),
                "word_count": ps.word_count,
                "dimensions": {
                    r.name: round(r.score, 3) for r in ps.scores
                },
            }
            for ps in report.paragraph_scores
        ]
    return data


def _display_paragraphs(report) -> None:
    """Display strongest and weakest paragraph sections."""
    if not report.paragraph_scores:
        console.print("[dim]No paragraphs long enough to score individually.[/dim]\n")
        return

    sorted_paras = sorted(report.paragraph_scores, key=lambda p: p.overall_score, reverse=True)
    top = sorted_paras[:3]
    bottom = sorted_paras[-3:] if len(sorted_paras) > 3 else []
    # Remove overlap if fewer than 6 paragraphs
    bottom = [p for p in bottom if p not in top]

    def _para_row(ps) -> str:
        dims = "  ".join(f"{r.name[:3]}={r.score:.2f}" for r in ps.scores)
        preview = ps.text_preview.replace("\n", " ")
        return f'  \u00b6{ps.index + 1:<3} "{preview[:55]:<55}"  {ps.overall_score:.2f}  {dims}'

    console.print("[bold green]Strongest sections:[/bold green]")
    for ps in top:
        console.print(_para_row(ps))

    if bottom:
        console.print("\n[bold red]Weakest sections:[/bold red]")
        for ps in reversed(bottom):
            console.print(_para_row(ps))

    console.print()


def _display_highlights(report) -> None:
    """Display matched highlights grouped by scorer."""
    has_any = False
    for result in report.scores:
        if not result.highlights:
            continue
        if not has_any:
            console.print("[bold]Highlights:[/bold]")
            has_any = True
        console.print(f"  [cyan]{result.name}:[/cyan]")
        for h in result.highlights[:10]:  # cap at 10 per scorer
            display_text = h.text[:50]
            console.print(f'    [{h.category:<15}] "{display_text}"  [dim](pos {h.position})[/dim]')
    if has_any:
        console.print()


def _report_to_dict_with_highlights(report, source: str | None = None) -> dict:
    """Convert a QualityReport to a JSON-serializable dict, including highlights."""
    data = _report_to_dict(report, source)
    for result in report.scores:
        if result.name in data["dimensions"]:
            data["dimensions"][result.name]["highlights"] = [
                {"text": h.text, "category": h.category, "position": h.position}
                for h in result.highlights
            ]
    return data


@main.command()
@click.argument("source")
@click.option("--scorers", "-s", help="Comma-separated scorer names", default=None)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--paragraphs", is_flag=True, help="Show per-paragraph breakdown")
@click.option("--highlights", is_flag=True, help="Show matched phrases per dimension")
@click.option("--profile", "-p", help="Scorer profile (default, technical, news, opinion)",
              default=None)
def score(source: str, scorers: str | None, as_json: bool, paragraphs: bool,
          highlights: bool, profile: str | None):
    """Score content quality from a URL or file.

    SOURCE can be a URL (https://...) or a file path (- for stdin).
    """
    scorer_names = scorers.split(",") if scorers else None

    label, text, metadata = _resolve_source(source)

    pipeline = Pipeline(scorers=scorer_names, profile=profile)
    report = pipeline.score(text, metadata=metadata, include_paragraphs=paragraphs)

    if as_json:
        import json

        if highlights:
            click.echo(json.dumps(_report_to_dict_with_highlights(report), indent=2))
        else:
            click.echo(json.dumps(_report_to_dict(report), indent=2))
    else:
        _display_report(report, source=label)
        if highlights:
            _display_highlights(report)
        if paragraphs:
            _display_paragraphs(report)


@main.command()
@click.argument("sources", nargs=-1)
@click.option("--from-file", "from_file", type=click.Path(exists=True),
              help="Read sources from a file (one per line)")
@click.option("--scorers", "-s", help="Comma-separated scorer names", default=None)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--profile", "-p", help="Scorer profile (default, technical, news, opinion)",
              default=None)
def batch(sources: tuple[str, ...], from_file: str | None, scorers: str | None, as_json: bool,
          profile: str | None):
    """Score multiple sources and compare them.

    Accepts multiple URLs or file paths as arguments.
    """
    all_sources = list(sources)
    if from_file:
        with open(from_file) as f:
            all_sources.extend(line.strip() for line in f if line.strip())

    if not all_sources:
        console.print("[red]No sources provided. Pass URLs/files as arguments or use --from-file.[/red]")
        raise SystemExit(1)

    scorer_names = scorers.split(",") if scorers else None

    # Resolve all sources
    texts: list[tuple[str, str]] = []
    metadata_list: list[dict | None] = []
    source_keys: list[str] = []
    for src in all_sources:
        label, text, meta = _resolve_source(src, quiet=not as_json)
        texts.append((label, text))
        metadata_list.append(meta)
        source_keys.append(src)

    # Score
    pipeline = Pipeline(scorers=scorer_names, profile=profile)
    results = pipeline.score_batch(texts, metadata=metadata_list)

    if as_json:
        import json

        data = [
            _report_to_dict(report, source=source_keys[i])
            for i, (label, report) in enumerate(results)
        ]
        click.echo(json.dumps(data, indent=2))
    else:
        # Show individual reports
        for label, report in results:
            _display_report(report, source=label)

        # Ranked summary table
        ranked = sorted(
            [(source_keys[i], label, report) for i, (label, report) in enumerate(results)],
            key=lambda x: x[2].overall_score,
            reverse=True,
        )

        table = Table(title="Ranked Summary", show_header=True, header_style="bold")
        table.add_column("Rank", style="bold", justify="right", width=4)
        table.add_column("Source", max_width=40)
        table.add_column("Overall", justify="right")
        table.add_column("Grade", justify="center")
        for sr in results[0][1].scores:
            table.add_column(sr.name.title(), justify="right")

        for rank, (src, label, report) in enumerate(ranked, 1):
            row = [
                str(rank),
                src[:40],
                f"{report.overall_score:.3f}",
                report.grade,
            ]
            for sr in report.scores:
                row.append(f"{sr.score:.3f}")
            table.add_row(*row)

        console.print(table)


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
def profiles():
    """List available scorer profiles."""
    from distill.profiles import list_profiles as _list_profiles

    table = Table(show_header=True, header_style="bold")
    table.add_column("Profile", style="cyan")
    table.add_column("Description")

    for name, desc in _list_profiles().items():
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
