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


def _report_to_dict(report, source: str | None = None,
                    include_highlights: bool = False) -> dict:
    """Convert a QualityReport to a JSON-serializable dict."""
    data = report.to_dict(include_highlights=include_highlights)
    if source is not None:
        data = {"source": source, **data}
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




@main.command()
@click.argument("source")
@click.option("--scorers", "-s", help="Comma-separated scorer names", default=None)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--csv", "as_csv", is_flag=True, help="Output as CSV")
@click.option("--paragraphs", is_flag=True, help="Show per-paragraph breakdown")
@click.option("--highlights", is_flag=True, help="Show matched phrases per dimension")
@click.option("--profile", "-p", help="Scorer profile (default, technical, news, opinion)",
              default=None)
@click.option("--auto-profile", is_flag=True, help="Auto-detect content type and select profile")
def score(source: str, scorers: str | None, as_json: bool, as_csv: bool, paragraphs: bool,
          highlights: bool, profile: str | None, auto_profile: bool):
    """Score content quality from a URL or file.

    SOURCE can be a URL (https://...) or a file path (- for stdin).
    """
    if as_json and as_csv:
        raise click.UsageError("--json and --csv are mutually exclusive.")
    if auto_profile and profile:
        raise click.UsageError("--auto-profile and --profile are mutually exclusive.")

    scorer_names = scorers.split(",") if scorers else None

    label, text, metadata = _resolve_source(source)

    pipeline = Pipeline(scorers=scorer_names, profile=profile, auto_profile=auto_profile)
    report = pipeline.score(text, metadata=metadata, include_paragraphs=paragraphs)

    if pipeline.detected_content_type and not as_json and not as_csv:
        ct = pipeline.detected_content_type
        console.print(f"[dim]Auto-detected: {ct.name} (confidence {ct.confidence:.2f})[/dim]")

    if as_json:
        import json

        data = _report_to_dict(report, include_highlights=highlights)
        if pipeline.detected_content_type:
            ct = pipeline.detected_content_type
            data["detected_type"] = ct.name
            data["detected_confidence"] = round(ct.confidence, 3)
        click.echo(json.dumps(data, indent=2))
    elif as_csv:
        from distill.export import report_to_csv_row, reports_to_csv

        row = report_to_csv_row(report, source=source)
        click.echo(reports_to_csv([row]), nl=False)
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
@click.option("--jsonl", "as_jsonl", is_flag=True, help="Output as newline-delimited JSON")
@click.option("--csv", "as_csv", is_flag=True, help="Output as CSV")
@click.option("--profile", "-p", help="Scorer profile (default, technical, news, opinion)",
              default=None)
@click.option("--auto-profile", is_flag=True, help="Auto-detect content type and select profile")
def batch(sources: tuple[str, ...], from_file: str | None, scorers: str | None, as_json: bool,
          as_jsonl: bool, as_csv: bool, profile: str | None, auto_profile: bool):
    """Score multiple sources and compare them.

    Accepts multiple URLs or file paths as arguments.
    """
    output_flags = sum([as_json, as_jsonl, as_csv])
    if output_flags > 1:
        raise click.UsageError("--json, --jsonl, and --csv are mutually exclusive.")
    if auto_profile and profile:
        raise click.UsageError("--auto-profile and --profile are mutually exclusive.")

    all_sources = list(sources)
    if from_file:
        with open(from_file) as f:
            all_sources.extend(line.strip() for line in f if line.strip())

    if not all_sources:
        console.print("[red]No sources provided. Pass URLs/files as arguments or use --from-file.[/red]")
        raise SystemExit(1)

    scorer_names = scorers.split(",") if scorers else None

    # Resolve all sources in parallel
    from concurrent.futures import ThreadPoolExecutor

    def _resolve(src):
        return (src, _resolve_source(src, quiet=True))

    texts: list[tuple[str, str]] = []
    metadata_list: list[dict | None] = []
    source_keys: list[str] = []

    max_workers = min(len(all_sources), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resolved = list(executor.map(_resolve, all_sources))

    for src, (label, text, meta) in resolved:
        texts.append((label, text))
        metadata_list.append(meta)
        source_keys.append(src)

    # Score
    pipeline = Pipeline(scorers=scorer_names, profile=profile, auto_profile=auto_profile)
    results = pipeline.score_batch(texts, metadata=metadata_list)

    if as_jsonl:
        from distill.export import report_to_jsonl_line

        for i, (label, report) in enumerate(results):
            click.echo(report_to_jsonl_line(report, source=source_keys[i]))
    elif as_json:
        import json

        data = [
            _report_to_dict(report, source=source_keys[i])
            for i, (label, report) in enumerate(results)
        ]
        click.echo(json.dumps(data, indent=2))
    elif as_csv:
        from distill.export import report_to_csv_row, reports_to_csv

        rows = [
            report_to_csv_row(report, source=source_keys[i])
            for i, (label, report) in enumerate(results)
        ]
        click.echo(reports_to_csv(rows), nl=False)
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
@click.argument("source_a")
@click.argument("source_b")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--profile", "-p", help="Scorer profile", default=None)
@click.option("--scorers", "-s", help="Comma-separated scorer names", default=None)
@click.option("--auto-profile", is_flag=True, help="Auto-detect content type and select profile")
def compare(source_a: str, source_b: str, as_json: bool, profile: str | None,
            scorers: str | None, auto_profile: bool):
    """Compare quality of two sources side by side.

    SOURCE_A and SOURCE_B can be URLs or file paths (- for stdin on one).
    """
    if auto_profile and profile:
        raise click.UsageError("--auto-profile and --profile are mutually exclusive.")

    scorer_names = scorers.split(",") if scorers else None

    label_a, text_a, meta_a = _resolve_source(source_a)
    label_b, text_b, meta_b = _resolve_source(source_b)

    pipeline = Pipeline(scorers=scorer_names, profile=profile, auto_profile=auto_profile)
    result = pipeline.compare(
        text_a, text_b,
        label_a=label_a, label_b=label_b,
        metadata_a=meta_a, metadata_b=meta_b,
    )

    if as_json:
        import json

        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _display_comparison(result, source_a, source_b)


def _display_comparison(result, source_a: str, source_b: str) -> None:
    """Rich display of a comparison result."""
    table = Table(title="Comparison", show_header=True, header_style="bold", box=None,
                  padding=(0, 2))
    table.add_column("Dimension", style="cyan")
    table.add_column(f"A: {source_a[:30]}", justify="right")
    table.add_column(f"B: {source_b[:30]}", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Winner", justify="center")

    for d in result.dimension_deltas:
        winner_str = ""
        if d.winner == "A":
            winner_str = "[green]<-- A[/green]"
        elif d.winner == "B":
            winner_str = "[green]B -->[/green]"
        else:
            winner_str = "[dim]tie[/dim]"

        delta_color = "green" if d.delta > 0 else ("red" if d.delta < 0 else "dim")
        delta_str = f"[{delta_color}]{d.delta:+.3f}[/{delta_color}]"

        table.add_row(
            d.name,
            f"{d.score_a:.3f}",
            f"{d.score_b:.3f}",
            delta_str,
            winner_str,
        )

    # Overall row
    table.add_row("", "", "", "", "")
    overall_delta = result.overall_delta
    delta_color = "green" if overall_delta > 0 else ("red" if overall_delta < 0 else "dim")
    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{result.report_a.overall_score:.3f}[/bold]",
        f"[bold]{result.report_b.overall_score:.3f}[/bold]",
        f"[bold {delta_color}]{overall_delta:+.3f}[/bold {delta_color}]",
        "",
    )

    console.print(table)

    # Winner banner
    if result.winner == "tie":
        console.print(
            Panel("[bold yellow]TIE[/bold yellow] — scores are within noise threshold",
                  border_style="yellow")
        )
    elif result.winner == "A":
        console.print(
            Panel(f"[bold green]WINNER: A[/bold green] ({result.label_a}) "
                  f"by {abs(overall_delta):.3f}",
                  border_style="green")
        )
    else:
        console.print(
            Panel(f"[bold green]WINNER: B[/bold green] ({result.label_b}) "
                  f"by {abs(overall_delta):.3f}",
                  border_style="green")
        )


@main.command()
@click.option("--port", default=7331, help="Port to listen on (default: 7331)")
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
def serve(port: int, host: str):
    """Start the scoring API server for the browser extension."""
    try:
        from distill.server import create_app
    except ImportError:
        console.print(
            "[red]Flask is required for the server. "
            "Install it with: pip install distill-score[server][/red]"
        )
        raise SystemExit(1)

    app = create_app()
    console.print(f"[bold]distill server[/bold] running on http://{host}:{port}")
    console.print("[dim]POST /score with {{\"html\": \"...\"}} or {{\"text\": \"...\"}}[/dim]")
    console.print("[dim]GET /health for status[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    app.run(host=host, port=port, debug=False)


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
