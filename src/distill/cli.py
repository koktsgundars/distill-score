"""CLI for distill content quality scoring."""

from __future__ import annotations

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
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # Skip if running under test harness or other non-standard environment

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
        ci_str = ""
        if result.ci_lower is not None and result.ci_upper is not None:
            ci_str = f" [{result.ci_lower:.2f}-{result.ci_upper:.2f}]"
        score_text = _score_bar(result.score)
        score_text.append(ci_str, style="dim")
        table.add_row(
            result.name,
            score_text,
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
        role_tag = f"[{ps.position_role}]"
        dims = "  ".join(f"{r.name[:3]}={r.score:.2f}" for r in ps.scores)
        preview = ps.text_preview.replace("\n", " ")
        return (
            f'  \u00b6{ps.index + 1:<3} {role_tag:<18}'
            f' "{preview[:50]:<50}"  {ps.overall_score:.2f}  {dims}'
        )

    console.print("[bold green]Strongest sections:[/bold green]")
    for ps in top:
        console.print(_para_row(ps))

    if bottom:
        console.print("\n[bold red]Weakest sections:[/bold red]")
        for ps in reversed(bottom):
            console.print(_para_row(ps))

    # Show weighted paragraph score
    wps = report.weighted_paragraph_score
    if wps is not None:
        console.print(f"\n  Weighted paragraph score: [bold]{wps:.3f}[/bold]")

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


def _display_report_from_dict(data: dict, source: str = "") -> None:
    """Rich display of a cached report dict (same layout as _display_report)."""
    grade = data.get("grade", "?")
    label = data.get("label", "")
    overall_score = data.get("overall_score", 0.0)
    word_count = data.get("word_count", 0)

    grade_colors = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red bold"}
    grade_style = grade_colors.get(grade, "white")

    title = "Quality Report"
    if source:
        title += f" - {source[:60]}"

    overall = Text()
    overall.append("\n  Grade: ", style="bold")
    overall.append(f"{grade}", style=f"bold {grade_style}")
    overall.append(f"  ({label})\n", style="dim")
    overall.append("  Overall: ")
    overall.append(_score_bar(overall_score))
    overall.append(f"\n  Words: {word_count:,}\n")

    console.print(Panel(overall, title=title, border_style="blue"))

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", min_width=25)
    table.add_column("Details", style="dim")

    for name, dim in data.get("dimensions", {}).items():
        ci_str = ""
        if "ci_lower" in dim and "ci_upper" in dim:
            ci_str = f" [{dim['ci_lower']:.2f}-{dim['ci_upper']:.2f}]"
        score_text = _score_bar(dim["score"])
        score_text.append(ci_str, style="dim")
        table.add_row(
            name,
            score_text,
            dim.get("explanation", "")[:80],
        )

    console.print(table)
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
@click.option("--no-cache", is_flag=True, help="Skip cache reads (still saves to history)")
def score(source: str, scorers: str | None, as_json: bool, as_csv: bool, paragraphs: bool,
          highlights: bool, profile: str | None, auto_profile: bool, no_cache: bool):
    """Score content quality from a URL or file.

    SOURCE can be a URL (https://...) or a file path (- for stdin).
    """
    if as_json and as_csv:
        raise click.UsageError("--json and --csv are mutually exclusive.")
    if auto_profile and profile:
        raise click.UsageError("--auto-profile and --profile are mutually exclusive.")

    scorer_names = scorers.split(",") if scorers else None

    label, text, metadata = _resolve_source(source)

    # Cache lookup
    from distill.cache import ScoreCache

    cache = ScoreCache()
    cached = None
    if not no_cache:
        cached = cache.get(text, profile=profile, scorer_names=scorer_names)

    if cached is not None:
        if not as_json and not as_csv:
            console.print("[dim]Using cached result[/dim]")
        if as_json:
            import json

            click.echo(json.dumps(cached, indent=2))
        elif as_csv:
            from distill.export import reports_to_csv

            row = {"source": source, **cached}
            click.echo(reports_to_csv([row]), nl=False)
        else:
            _display_report_from_dict(cached, source=label)
        return

    pipeline = Pipeline(scorers=scorer_names, profile=profile, auto_profile=auto_profile)
    report = pipeline.score(text, metadata=metadata, include_paragraphs=paragraphs)

    # Save to cache
    effective_scorer_names = scorer_names or [s.name for s in pipeline._scorers]
    report_dict = report.to_dict(include_highlights=True)
    cache.put(text, report_dict, source=source, profile=profile,
              scorer_names=effective_scorer_names, metadata=metadata)

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
@click.option("--no-cache", is_flag=True, help="Skip cache reads (still saves to history)")
def batch(sources: tuple[str, ...], from_file: str | None, scorers: str | None, as_json: bool,
          as_jsonl: bool, as_csv: bool, profile: str | None, auto_profile: bool,
          no_cache: bool):
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

    # Cache: check for hits per item
    from distill.cache import ScoreCache

    cache = ScoreCache()

    # Score (with per-item cache check)
    pipeline = Pipeline(scorers=scorer_names, profile=profile, auto_profile=auto_profile)
    effective_scorer_names = scorer_names or list(
        s.name for s in pipeline._scorers
    )

    results: list[tuple[str, object]] = []
    to_score_indices: list[int] = []
    to_score_texts: list[tuple[str, str]] = []
    to_score_metadata: list[dict | None] = []

    for i, (label_text, meta) in enumerate(zip(texts, metadata_list)):
        label, text = label_text
        if not no_cache:
            cached = cache.get(text, profile=profile, scorer_names=effective_scorer_names)
            if cached is not None:
                results.append((i, label, cached, True))  # type: ignore[arg-type]
                continue
        to_score_indices.append(i)
        to_score_texts.append((label, text))
        to_score_metadata.append(meta)

    if to_score_texts:
        scored = pipeline.score_batch(to_score_texts, metadata=to_score_metadata)
        for j, (label, report) in enumerate(scored):
            idx = to_score_indices[j]
            text = texts[idx][1]
            report_dict = report.to_dict(include_highlights=True)
            cache.put(text, report_dict, source=source_keys[idx], profile=profile,
                      scorer_names=effective_scorer_names, metadata=metadata_list[idx])
            results.append((idx, label, report, False))  # type: ignore[arg-type]

    # Sort back to original order
    results.sort(key=lambda x: x[0])  # type: ignore[index]

    # Unpack — for cached items we have dicts, for scored items we have QualityReport
    final_results: list[tuple[str, object, bool]] = [
        (label, data, is_cached) for _, label, data, is_cached in results  # type: ignore[misc]
    ]

    if as_jsonl:
        import json as json_mod

        for i, (label, data, is_cached) in enumerate(final_results):
            if is_cached:
                row_data = {"source": source_keys[i], **data}
                click.echo(json_mod.dumps(row_data))
            else:
                from distill.export import report_to_jsonl_line
                click.echo(report_to_jsonl_line(data, source=source_keys[i]))
    elif as_json:
        import json

        out = []
        for i, (label, data, is_cached) in enumerate(final_results):
            if is_cached:
                out.append({"source": source_keys[i], **data})
            else:
                out.append(_report_to_dict(data, source=source_keys[i]))
        click.echo(json.dumps(out, indent=2))
    elif as_csv:
        from distill.export import report_to_csv_row, reports_to_csv

        rows = []
        for i, (label, data, is_cached) in enumerate(final_results):
            if is_cached:
                rows.append({"source": source_keys[i], **data})
            else:
                rows.append(report_to_csv_row(data, source=source_keys[i]))
        click.echo(reports_to_csv(rows), nl=False)
    else:
        # Show individual reports
        for i, (label, data, is_cached) in enumerate(final_results):
            if is_cached:
                _display_report_from_dict(data, source=label)
            else:
                _display_report(data, source=label)

        # Ranked summary table — normalize to dicts for uniform access
        ranked_data = []
        for i, (label, data, is_cached) in enumerate(final_results):
            if is_cached:
                d = data
            else:
                d = data.to_dict()
            ranked_data.append((source_keys[i], label, d))

        ranked_data.sort(key=lambda x: x[2]["overall_score"], reverse=True)

        table = Table(title="Ranked Summary", show_header=True, header_style="bold")
        table.add_column("Rank", style="bold", justify="right", width=4)
        table.add_column("Source", max_width=40)
        table.add_column("Overall", justify="right")
        table.add_column("Grade", justify="center")
        first_dims = ranked_data[0][2].get("dimensions", {})
        for dim_name in first_dims:
            table.add_column(dim_name.title(), justify="right")

        for rank, (src, label, d) in enumerate(ranked_data, 1):
            row = [
                str(rank),
                src[:40],
                f"{d['overall_score']:.3f}",
                d["grade"],
            ]
            for dim_name, dim_data in d.get("dimensions", {}).items():
                row.append(f"{dim_data['score']:.3f}")
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


@main.group()
def history():
    """Browse and manage score history."""
    pass


@history.command(name="show")
@click.option("--limit", "-n", default=20, help="Number of entries to show (default: 20)")
@click.option("--source", help="Filter by source substring")
def history_show(limit: int, source: str | None):
    """Show recent scoring history."""
    from distill.cache import ScoreCache

    cache = ScoreCache()
    entries = cache.history(source=source, limit=limit)

    if not entries:
        console.print("[dim]No history entries found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", justify="right")
    table.add_column("Source", max_width=40)
    table.add_column("Profile", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Grade", justify="center")
    table.add_column("Words", justify="right")
    table.add_column("Scored At", style="dim")

    for entry in entries:
        table.add_row(
            str(entry["id"]),
            (entry["source"] or "")[:40],
            entry["profile"],
            f"{entry['overall_score']:.3f}",
            entry["grade"],
            str(entry["word_count"]),
            entry["scored_at"][:19],
        )

    console.print(table)


@history.command(name="clear")
@click.option("--before", help="Delete entries scored before this ISO date")
@click.option("--source", help="Filter by source substring")
@click.confirmation_option(prompt="Are you sure you want to delete history entries?")
def history_clear(before: str | None, source: str | None):
    """Clear scoring history."""
    from distill.cache import ScoreCache

    cache = ScoreCache()
    deleted = cache.clear(before=before, source=source)
    console.print(f"Deleted {deleted} history entries.")


@history.command(name="stats")
def history_stats():
    """Show cache statistics."""
    from distill.cache import ScoreCache

    cache = ScoreCache()
    stats = cache.stats()

    console.print("[bold]Cache Statistics[/bold]")
    console.print(f"  Entries:  {stats['count']}")
    console.print(f"  DB size:  {stats['size_bytes']:,} bytes")
    console.print(f"  Oldest:   {stats['oldest'] or 'N/A'}")
    console.print(f"  Newest:   {stats['newest'] or 'N/A'}")


@history.command(name="export")
@click.option("--json", "as_json", is_flag=True, help="Export as JSON")
@click.option("--csv", "as_csv", is_flag=True, help="Export as CSV")
@click.option("--limit", "-n", default=100, help="Maximum entries to export (default: 100)")
@click.option("--source", help="Filter by source substring")
def history_export(as_json: bool, as_csv: bool, limit: int, source: str | None):
    """Export scoring history."""
    import json as json_mod

    if as_json and as_csv:
        raise click.UsageError("--json and --csv are mutually exclusive.")

    from distill.cache import ScoreCache

    cache = ScoreCache()
    entries = cache.history(source=source, limit=limit)

    if not entries:
        console.print("[dim]No history entries to export.[/dim]")
        return

    if as_csv:
        import csv
        import io

        buf = io.StringIO()
        fieldnames = ["id", "source", "profile", "overall_score", "grade", "word_count",
                       "scored_at"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({k: entry[k] for k in fieldnames})
        click.echo(buf.getvalue(), nl=False)
    else:
        # Default to JSON
        click.echo(json_mod.dumps(entries, indent=2))


_GRADE_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}


@main.command()
@click.argument("sources", nargs=-1)
@click.option("--from-file", "from_file", type=click.Path(exists=True),
              help="Read sources from a file (one per line)")
@click.option("--min-grade", default="C",
              type=click.Choice(["A", "B", "C", "D", "F"], case_sensitive=False),
              help="Minimum acceptable grade (default: C)")
@click.option("--min-score", default=None, type=float,
              help="Minimum acceptable overall score (0.0-1.0). Overrides --min-grade.")
@click.option("--scorers", "-s", help="Comma-separated scorer names", default=None)
@click.option("--json", "as_json", is_flag=True, help="Output JSON with pass/fail per source")
@click.option("--profile", "-p", help="Scorer profile (default, technical, news, opinion)",
              default=None)
@click.option("--auto-profile", is_flag=True, help="Auto-detect content type and select profile")
@click.option("--no-cache", is_flag=True, help="Skip cache reads (still saves to history)")
def gate(sources: tuple[str, ...], from_file: str | None, min_grade: str, min_score: float | None,
         scorers: str | None, as_json: bool, profile: str | None, auto_profile: bool,
         no_cache: bool):
    """Quality gate — fail if content is below threshold.

    Exits 0 if all sources pass, exits 1 if any fail.
    Useful for CI pipelines and pre-commit hooks.
    """
    if auto_profile and profile:
        raise click.UsageError("--auto-profile and --profile are mutually exclusive.")

    all_sources = list(sources)
    if from_file:
        with open(from_file) as f:
            all_sources.extend(line.strip() for line in f if line.strip())

    if not all_sources:
        console.print("[red]No sources provided. Pass file paths/URLs as arguments "
                       "or use --from-file.[/red]")
        raise SystemExit(1)

    scorer_names = scorers.split(",") if scorers else None
    min_grade_upper = min_grade.upper()

    pipeline = Pipeline(scorers=scorer_names, profile=profile, auto_profile=auto_profile)

    # Score each source
    gate_results: list[dict] = []
    any_failed = False

    for src in all_sources:
        label, text, metadata = _resolve_source(src, quiet=True)

        # Cache check
        report = None
        if not no_cache:
            from distill.cache import ScoreCache

            cache = ScoreCache()
            cached = cache.get(text, profile=profile, scorer_names=scorer_names)
            if cached is not None:
                score_val = cached["overall_score"]
                grade_val = cached["grade"]
            else:
                cached = None

        if not no_cache and cached is not None:
            pass  # score_val and grade_val already set
        else:
            report = pipeline.score(text, metadata=metadata)
            score_val = report.overall_score
            grade_val = report.grade

            # Save to cache
            from distill.cache import ScoreCache

            cache = ScoreCache()
            effective = scorer_names or [s.name for s in pipeline._scorers]
            report_dict = report.to_dict(include_highlights=False)
            cache.put(text, report_dict, source=src, profile=profile,
                      scorer_names=effective, metadata=metadata)

        # Determine pass/fail
        if min_score is not None:
            passed = score_val >= min_score
        else:
            passed = _GRADE_ORDER.get(grade_val, 0) >= _GRADE_ORDER.get(min_grade_upper, 2)

        if not passed:
            any_failed = True

        gate_results.append({
            "source": src,
            "score": round(score_val, 3),
            "grade": grade_val,
            "passed": passed,
        })

    if as_json:
        import json

        threshold = {"min_score": min_score} if min_score is not None else {"min_grade": min_grade_upper}
        output = {
            "threshold": threshold,
            "all_passed": not any_failed,
            "results": gate_results,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Compact table
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Source", max_width=50)
        table.add_column("Score", justify="right")
        table.add_column("Grade", justify="center")
        table.add_column("Status", justify="center")

        for r in gate_results:
            status = "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]"
            table.add_row(r["source"][:50], f"{r['score']:.3f}", r["grade"], status)

        console.print(table)

        if any_failed:
            threshold_str = (f"score >= {min_score}" if min_score is not None
                             else f"grade >= {min_grade_upper}")
            console.print(f"\n[red bold]FAILED[/red bold] — "
                          f"threshold: {threshold_str}")
        else:
            console.print("\n[green bold]PASSED[/green bold] — all sources meet quality gate")

    if any_failed:
        raise SystemExit(1)


@main.command()
@click.argument("source")
@click.option("--scorers", "-s", help="Comma-separated scorer names", default=None)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--paragraphs", is_flag=True, help="Show per-paragraph breakdown")
@click.option("--highlights", is_flag=True, help="Show matched phrases per dimension")
@click.option("--profile", "-p", help="Scorer profile (default, technical, news, opinion)",
              default=None)
@click.option("--auto-profile", is_flag=True, help="Auto-detect content type and select profile")
@click.option("--no-cache", is_flag=True, help="Skip cache reads (still saves to history)")
@click.option("--debounce", default=2.0, type=float,
              help="Seconds to wait after last change before re-scoring (default: 2.0)")
def watch(source: str, scorers: str | None, as_json: bool, paragraphs: bool,
          highlights: bool, profile: str | None, auto_profile: bool, no_cache: bool,
          debounce: float):
    """Watch a file and re-score on changes.

    SOURCE must be a file path (URLs and stdin are not supported).
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        console.print(
            "[red]watchdog is required for watch mode. "
            "Install it with: pip install distill-score[watch][/red]"
        )
        raise SystemExit(1)

    if source == "-":
        raise click.UsageError("Watch mode does not support stdin. Provide a file path.")
    if source.startswith(("http://", "https://")):
        raise click.UsageError("Watch mode does not support URLs. Provide a file path.")

    import os

    filepath = os.path.abspath(source)
    if not os.path.isfile(filepath):
        console.print(f"[red]File not found: {source}[/red]")
        raise SystemExit(1)

    if auto_profile and profile:
        raise click.UsageError("--auto-profile and --profile are mutually exclusive.")

    scorer_names = scorers.split(",") if scorers else None

    import threading

    def _score_and_display():
        """Read the file, score it, and display the report."""
        try:
            with open(filepath) as f:
                text = f.read()
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            return

        if not text.strip():
            console.print("[dim]File is empty, skipping.[/dim]")
            return

        pipeline = Pipeline(scorers=scorer_names, profile=profile, auto_profile=auto_profile)
        report = pipeline.score(text, include_paragraphs=paragraphs)

        # Cache
        if not no_cache:
            from distill.cache import ScoreCache

            cache = ScoreCache()
            effective = scorer_names or [s.name for s in pipeline._scorers]
            report_dict = report.to_dict(include_highlights=True)
            cache.put(text, report_dict, source=source, profile=profile,
                      scorer_names=effective)

        if pipeline.detected_content_type and not as_json:
            ct = pipeline.detected_content_type
            console.print(f"[dim]Auto-detected: {ct.name} (confidence {ct.confidence:.2f})[/dim]")

        if as_json:
            import json

            data = _report_to_dict(report, include_highlights=highlights)
            click.echo(json.dumps(data, indent=2))
        else:
            _display_report(report, source=source)
            if highlights:
                _display_highlights(report)
            if paragraphs:
                _display_paragraphs(report)

    # Initial score
    console.print(f"[bold]Watching[/bold] {source} [dim](debounce {debounce}s, Ctrl+C to stop)[/dim]\n")
    _score_and_display()

    # Debounced file watcher
    timer: threading.Timer | None = None
    timer_lock = threading.Lock()

    def _on_change():
        """Called after debounce delay."""
        # Clear screen and re-score
        click.clear()
        console.print(f"[bold]Watching[/bold] {source} [dim](debounce {debounce}s, Ctrl+C to stop)[/dim]\n")
        _score_and_display()

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if os.path.abspath(event.src_path) != filepath:
                return
            nonlocal timer
            with timer_lock:
                if timer is not None:
                    timer.cancel()
                timer = threading.Timer(debounce, _on_change)
                timer.daemon = True
                timer.start()

    observer = Observer()
    observer.schedule(_Handler(), os.path.dirname(filepath), recursive=False)
    observer.start()

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping watch...[/dim]")
    finally:
        observer.stop()
        observer.join()
        with timer_lock:
            if timer is not None:
                timer.cancel()


if __name__ == "__main__":
    main()
