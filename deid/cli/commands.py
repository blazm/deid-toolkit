"""Flat commands for deid: list <target>, select <target> <items>, run <stage>.

Command layout
--------------
deid list datasets|techniques|evaluation|results
deid select datasets|techniques|evaluation  <names or indices>
deid run all|preprocess|techniques|evaluation|logs
deid show
deid migrate [--yes]
deid explore [--port N]
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import typer

# Three Typer groups: list / select / run
list_app = typer.Typer(help="List available datasets, techniques, evaluations, results.")
select_app = typer.Typer(help="Select datasets, techniques, or evaluation metrics.")
run_app = typer.Typer(help="Run a pipeline stage or the full pipeline.")

# -------------- --------------------------------------------------
# list commands
# -------------- --------------------------------------------------

@list_app.command()
def datasets() -> None:
    """List available datasets."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    original = set(loader.load_datasets())
    aligned = set(loader.load_aligned_datasets())
    all_names = sorted(original | aligned)
    if not all_names:
        typer.echo("No datasets found.")
        return
    for i, name in enumerate(all_names):
        marker = " [aligned]" if name in aligned else ""
        typer.echo(f"  {i}. {name}{marker}")


@list_app.command()
def techniques() -> None:
    """List available techniques."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    techs = loader.load_techniques()
    renames = loader.settings.technique_renames
    if not techs:
        typer.echo("No techniques found.")
        return
    for i, name in enumerate(techs):
        rename = renames.get(name, name)
        typer.echo(f"  {i}. {name}  ({rename})")


@list_app.command()
def evaluation() -> None:
    """List available evaluation metrics."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    evals = loader.load_evaluations()
    renames = loader.settings.evaluation_renames
    if not evals:
        typer.echo("No evaluation methods found.")
        return
    for i, name in enumerate(evals):
        rename = renames.get(name, name)
        typer.echo(f"  {i}. {name}  ({rename})")


@list_app.command()
def results() -> None:
    """List available evaluation results."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    results = loader.list_results()
    if not results:
        typer.echo("No results found.")
        return
    for dataset, techniques in sorted(results.items()):
        typer.echo(f"\n  {dataset}:")
        for technique, metrics in sorted(techniques.items()):
            typer.echo(f"    {technique}: {', '.join(metrics.keys())}")


@list_app.command()
def selected() -> None:
    """Show what would be run by ``deid run selected``.

    Displays selections and which stages are already complete.
    """
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    s = loader.settings

    typer.echo("Current selections:")
    typer.echo(f"  datasets:       {', '.join(s.datasets.selected) or '(none)'}")
    typer.echo(f"  techniques:     {', '.join(s.techniques.selected) or '(none)'}")
    typer.echo(f"  evaluations:    {', '.join(s.evaluation.selected) or '(none)'}")

    results_dir = loader.settings.result_path
    if not results_dir.is_dir():
        typer.echo("\nNo results directory found.")
        typer.echo("All stages would run: preprocess, techniques, evaluation")
        return

    ds_set = set(s.datasets.selected)
    completed_ds = set()
    completed_tech: dict[str, set[str]] = {ds: set() for ds in ds_set}
    completed_eval: dict[str, dict[str, set[str]]] = {ds: {t: set() for t in s.techniques.selected} for ds in ds_set}

    # Walk {technique}/{dataset}/*.csv hierarchy
    for tech_dir in results_dir.iterdir():
        if not tech_dir.is_dir() or tech_dir.name == "__pycache__":
            continue
        tech_name = tech_dir.name
        for ds_dir in tech_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            ds_name = ds_dir.name
            if ds_name not in ds_set:
                continue
            completed_ds.add(ds_name)
            completed_tech[ds_name].add(tech_name)
            for csv in ds_dir.glob("*.csv"):
                completed_eval.setdefault(ds_name, {}).setdefault(tech_name, set()).add(csv.stem)

    typer.echo("\nCompletion status:")
    for ds in sorted(ds_set):
        status = "  [done] " if ds in completed_ds else "  [skip] "
        techs_done = sorted(completed_tech.get(ds, set()))
        all_techs = set(s.techniques.selected)
        remaining_techs = all_techs - set(techs_done)
        if remaining_techs:
            typer.echo(f"{status}{ds}  (techniques done: {', '.join(techs_done)})")
        else:
            ev_done = sorted(completed_eval.get(ds, {}).get(all_techs[0] if all_techs else "", set()))
            all_evs = set(s.evaluation.selected)
            remaining_evs = all_evs - set(ev_done)
            if remaining_evs:
                typer.echo(f"{status}{ds}  (evaluations done: {', '.join(ev_done)})")
            else:
                typer.echo(f"{status}{ds}  (all complete)")

    stages_to_run: list[str] = []
    if not completed_ds:
        stages_to_run.append("preprocess")
    if any(set(s.techniques.selected) - completed_tech.get(ds, set()) for ds in ds_set):
        stages_to_run.append("techniques")
    if any(set(s.evaluation.selected) - set().union(*[set().union(v for v in completed_eval.get(ds, {}).values())]) for ds in ds_set):
        stages_to_run.append("evaluation")

    if not stages_to_run:
        typer.echo("\nNothing selected or nothing to run.")
    else:
        typer.echo(f"\n``deid run selected`` would run: {', '.join(stages_to_run)}")


# -------------- --------------------------------------------------
# select commands
# -------------- --------------------------------------------------


def _resolve_items(names: list[str], all_names: list[str], label: str) -> list[str]:
    """Resolve a list of names or indices to actual names."""
    selected: list[str] = []
    seen: set[str] = set()
    for item in names:
        try:
            idx = int(item)
            if idx < 0 or idx >= len(all_names):
                typer.echo(f"  {label} out of range: {item} (0-{len(all_names) - 1})")
                continue
            actual = all_names[idx]
        except ValueError:
            if item in all_names:
                actual = item
            else:
                typer.echo(f"  Unknown {label}: {item}")
                continue
        if actual not in seen:
            selected.append(actual)
            seen.add(actual)
    return selected


@select_app.command()
def datasets(names: str) -> None:
    """Select datasets by name or index.

    Usage:  deid select datasets 0 1        (by index)
            deid select datasets arface lfw  (by name)
    """
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    # Prefer aligned datasets (most users have pre-aligned data)
    all_names = loader.load_aligned_datasets() or loader.load_datasets()
    if not all_names:
        typer.echo("No datasets available.")
        return
    items = [n.strip() for n in names.replace(",", " ").split() if n.strip()]
    selected = _resolve_items(items, all_names, "dataset")
    if not selected:
        return
    _save_selection(loader, datasets=selected)
    typer.echo(f"Selected datasets: {', '.join(selected)}")


@select_app.command()
def techniques(names: str) -> None:
    """Select techniques by name or index."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    all_names = loader.load_techniques()
    if not all_names:
        typer.echo("No techniques available.")
        return
    items = [n.strip() for n in names.replace(",", " ").split() if n.strip()]
    selected = _resolve_items(items, all_names, "technique")
    if not selected:
        return
    _save_selection(loader, techniques=selected)
    typer.echo(f"Selected techniques: {', '.join(selected)}")


@select_app.command()
def evaluation(names: str) -> None:
    """Select evaluation metrics by name or index."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    all_names = loader.load_evaluations()
    if not all_names:
        typer.echo("No evaluation methods available.")
        return
    items = [n.strip() for n in names.replace(",", " ").split() if n.strip()]
    selected = _resolve_items(items, all_names, "evaluation")
    if not selected:
        return
    _save_selection(loader, evaluation=selected)
    typer.echo(f"Selected evaluations: {', '.join(selected)}")


@select_app.command()
def all(
    ds: str = typer.Option("", "-d", "--ds", help="Comma-separated dataset names or indices"),
    tech: str = typer.Option("", "-t", "--tech", help="Comma-separated technique names or indices"),
    eval: str = typer.Option("", "-e", "--eval", help="Comma-separated evaluation names or indices"),  # noqa: A002
) -> None:
    """Select datasets, techniques, and evaluations in one command."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    all_ds = loader.load_aligned_datasets()
    all_tech = loader.load_techniques()
    all_eval = loader.load_evaluations()

    ds_selected = _parse_selection(ds, all_ds, "dataset") if ds else []
    tech_selected = _parse_selection(tech, all_tech, "technique") if tech else []
    eval_selected = _parse_selection(eval, all_eval, "evaluation") if eval else []

    if not ds_selected and not tech_selected and not eval_selected:
        typer.echo("Nothing specified. Use 'deid select wizard' for guided selection.")
        return

    if not tech_selected:
        tech_selected = []  # Validation is always automatic

    _save_selection(loader, datasets=ds_selected, techniques=tech_selected, evaluation=eval_selected)

    typer.echo(f"Selected datasets:       {', '.join(ds_selected) or '(none)'}")
    typer.echo(f"Selected techniques:     {', '.join(tech_selected) or '(none)'} (validation is automatic)")
    typer.echo(f"Selected evaluations:    {', '.join(eval_selected) or '(none)'}")


@select_app.command()
def wizard() -> None:
    """Interactive wizard: select datasets, techniques, evaluations, then run or save."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()

    all_ds = loader.load_aligned_datasets()
    all_tech = loader.load_techniques()
    all_eval = loader.load_evaluations()

    # Clear previous selections for a clean slate
    config_path = loader._yaml_path
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("datasets"):
            if isinstance(cfg["datasets"], dict):
                cfg["datasets"]["selected"] = []
            else:
                cfg["datasets"] = {"selected": []}
        if cfg.get("techniques"):
            if isinstance(cfg["techniques"], dict):
                cfg["techniques"]["selected"] = []
            else:
                cfg["techniques"] = {"selected": []}
        if cfg.get("evaluation"):
            if isinstance(cfg["evaluation"], dict):
                cfg["evaluation"]["selected"] = []
            else:
                cfg["evaluation"] = {"selected": []}
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        typer.echo("Cleared previous selections for a fresh start.")

    # --- Datasets ---
    typer.echo()
    typer.echo("=== DEID Toolkit Wizard ===")
    typer.echo()

    typer.echo("Available datasets:")
    for i, name in enumerate(all_ds):
        typer.echo(f"  [ ] {i}. {name}")
    typer.echo()

    ds_input = typer.prompt("Select datasets (indices or names, space/comma separated)").strip()
    ds_selected = _parse_selection(ds_input, all_ds, "dataset") if ds_input else []

    # --- Techniques ---
    typer.echo()
    typer.echo("Available techniques (validation is always automatic):")
    for i, name in enumerate(all_tech):
        tag = " [auto]" if name == "validation" else ""
        typer.echo(f"  [ ] {i}. {name}{tag}")
    typer.echo()

    tech_input = typer.prompt("Select techniques (indices or names, space/comma separated)").strip()
    tech_selected = _parse_selection(tech_input, all_tech, "technique") if tech_input else []

    # --- Evaluations ---
    typer.echo()
    typer.echo("Available evaluations:")
    for i, name in enumerate(all_eval):
        typer.echo(f"  [ ] {i}. {name}")
    typer.echo()

    eval_input = typer.prompt("Select evaluations (indices or names, space/comma separated)").strip()
    eval_selected = _parse_selection(eval_input, all_eval, "evaluation") if eval_input else []

    # --- Summary ---
    typer.echo()
    typer.echo("=== Your Selection ===")
    typer.echo(f"  datasets:       {', '.join(ds_selected) or '(none)'}")
    typer.echo(f"  techniques:     {', '.join(tech_selected) or '(none)'}")
    typer.echo(f"  evaluations:    {', '.join(eval_selected) or '(none)'}")
    typer.echo()

    run = typer.confirm("Run pipeline now?", default=True)
    if run:
        _save_selection(loader, datasets=ds_selected, techniques=tech_selected, evaluation=eval_selected)
        _run_selected(loader)
    else:
        save = typer.confirm("Save for later?", default=True)
        if save:
            _save_selection(loader, datasets=ds_selected, techniques=tech_selected, evaluation=eval_selected)
            typer.echo("Selection saved. Run later with: deid run selected")
        else:
            typer.echo("Changes discarded.")


# -------------- --------------------------------------------------
# Helper functions
# -------------- --------------------------------------------------


def _parse_selection(input_str: str, all_names: list[str], _label: str) -> list[str]:
    """Parse user input: comma-separated, space-separated, or bracket ranges."""
    selected: list[str] = []
    seen: set[str] = set()

    # Normalize: replace commas/spaces, strip brackets
    input_str = input_str.strip("[] ").replace(",,", ",")
    # Split by comma or whitespace
    parts = []
    for ch in input_str:
        if ch in (",", " ", "\t", "\n", "\r"):
            parts.append(" ")
        else:
            parts.append(ch)
    tokens = "".join(parts).split()

    for token in tokens:
        if not token:
            continue
        # Check for ranges like "0-3" or "0..3"
        if "-" in token and token.replace("-", "").replace("0", "").isdigit():
            parts_r = token.split("-")
            if len(parts_r) == 2:
                try:
                    start, end = int(parts_r[0]), int(parts_r[1])
                    for i in range(start, end + 1):
                        if 0 <= i < len(all_names) and all_names[i] not in seen:
                            selected.append(all_names[i])
                            seen.add(all_names[i])
                    continue
                except (ValueError, IndexError):
                    pass
        # Single item: index or name
        try:
            idx = int(token)
            if 0 <= idx < len(all_names) and all_names[idx] not in seen:
                selected.append(all_names[idx])
                seen.add(all_names[idx])
        except ValueError:
            if token in all_names and token not in seen:
                selected.append(token)
                seen.add(token)
    return selected


def _save_selection(
    loader: "ConfigLoader",
    datasets: list[str] | None = None,
    techniques: list[str] | None = None,
    evaluation: list[str] | None = None,
) -> None:
    """Persist selections back to deid-config.yaml."""
    import yaml

    config_path = Path("root_dir/deid-config.yaml")
    if not config_path.exists():
        config_path = loader._yaml_path  # try default
    if not config_path.exists():
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if datasets is not None:
        cfg.setdefault("datasets", {}).setdefault("selected", [])
        cfg["datasets"]["selected"] = datasets
    if techniques is not None:
        cfg.setdefault("techniques", {})
        cfg.setdefault("techniques", {}).setdefault("selected", [])
        if "selected" in cfg.get("techniques", {}):
            cfg["techniques"]["selected"] = techniques
        else:
            cfg["techniques"] = {"selected": techniques}
    if evaluation is not None:
        cfg.setdefault("evaluation", {})
        if "selected" in cfg.get("evaluation", {}):
            cfg["evaluation"]["selected"] = evaluation
        else:
            cfg["evaluation"] = {"selected": evaluation}

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Invalidate loader's cached settings so the next read picks up the new values
    if hasattr(loader, "_settings"):
        loader._settings = None


def _run_selected(loader: "ConfigLoader") -> None:
    """Run the selected pipeline stages."""
    from deid.pipeline import run_preprocess, run_techniques, run_evaluations

    s = loader.settings
    if not s.datasets.selected or not s.evaluation.selected:
        typer.echo("Need at least datasets and evaluations selected.")
        raise typer.Exit(1)

    # Exclude validation from explicit technique list
    techs = [t for t in s.techniques.selected if t != "validation"]
    if not techs:
        typer.echo("No technique selected — only validation will run.")

    typer.echo()
    typer.echo("=== Running Pipeline ===")

    typer.echo("\n[1/3] Preprocessing...")
    success1 = run_preprocess(loader)
    typer.echo("  OK" if success1 else "  FAIL")

    if techs:
        typer.echo("\n[2/3] Techniques...")
        s.techniques.selected = techs
        success2 = run_techniques(loader)
        typer.echo("  OK" if success2 else "  FAIL")
    else:
        typer.echo("\n[2/3] Techniques — skipped (validation only, runs automatically)")
        success2 = True

    typer.echo("\n[3/3] Evaluations...")
    success3 = run_evaluations(loader)
    typer.echo("  OK" if success3 else "  FAIL")

    typer.echo()
    if success1 and success2 and success3:
        typer.echo("=== Pipeline complete! ===")
    else:
        typer.echo("=== Pipeline finished with errors. Check logs. ===")
    raise typer.Exit(0)


# -------------- --------------------------------------------------
# run commands
# -------------- --------------------------------------------------

@run_app.command()
def all() -> None:
    """Execute the full pipeline: preprocess -> techniques -> evaluations."""
    from deid.pipeline import run_all
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    s = loader.settings
    if not s.datasets.selected or not s.techniques.selected or not s.evaluation.selected:
        typer.echo("No datasets, techniques, or evaluations selected.")
        raise typer.Exit(1)
    success = run_all(loader)
    typer.echo("Pipeline complete." if success else "Pipeline finished with errors.")
    raise typer.Exit(0 if success else 1)


@run_app.command()
def preprocess() -> None:
    """Run preprocessing only (alignment + pair generation)."""
    from deid.pipeline import run_preprocess
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    success = run_preprocess(loader)
    typer.echo("Preprocessing complete." if success else "Preprocessing failed.")
    raise typer.Exit(0 if success else 1)


@run_app.command()
def techniques() -> None:
    """Run techniques only."""
    from deid.pipeline import run_techniques
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    success = run_techniques(loader)
    typer.echo("Techniques complete." if success else "Techniques failed.")
    raise typer.Exit(0 if success else 1)


@run_app.command()
def evaluation() -> None:
    """Run evaluations only."""
    from deid.pipeline import run_evaluations
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    success = run_evaluations(loader)
    typer.echo("Evaluations complete." if success else "Evaluations failed.")
    raise typer.Exit(0 if success else 1)


@run_app.command()
def validation() -> None:
    """Run preprocessing + evaluation using only validation (aligned images as reference)."""
    from deid.pipeline import run_preprocess, run_evaluations
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    s = loader.settings

    ds_selected = s.datasets.selected
    eval_selected = s.evaluation.selected
    if not ds_selected or not eval_selected:
        typer.echo("No datasets or evaluations selected. Use 'deid select <target> <items>' first.")
        raise typer.Exit(1)

    s.techniques.selected = []  # No real technique selection needed

    typer.echo("Running preprocessing...")
    success = run_preprocess(loader)
    typer.echo("  preprocess OK" if success else "  preprocess FAILED")

    typer.echo("Running validation evaluation...")
    success = run_evaluations(loader)
    typer.echo("  evaluation OK" if success else "  evaluation FAILED")

    typer.echo("Validation complete.")


@run_app.command()
def logs() -> None:
    """Show the latest pipeline log."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    logs_dir = str(loader.settings.logs_path)
    if not os.path.isdir(logs_dir):
        typer.echo("No logs directory found.")
        return
    files = sorted(os.listdir(logs_dir), reverse=True)
    if not files:
        typer.echo("No log files found.")
        return
    latest = os.path.join(logs_dir, files[0])
    typer.echo(f"Tail of {latest}:")
    with open(latest) as f:
        lines = f.readlines()
        for line in lines[-50:]:
            typer.echo(line.rstrip())


@run_app.command()
def selected() -> None:
    """Run only the stages that haven't been completed yet.

    Checks existing results in root_dir/results/ and skips stages
    that already have output. Useful for resuming partial runs.
    """
    from deid.config.loader import ConfigLoader
    from deid.pipeline import run_preprocess, run_techniques, run_evaluations
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    loader = ConfigLoader()
    s = loader.settings

    ds_selected = s.datasets.selected
    tech_selected = s.techniques.selected
    eval_selected = s.evaluation.selected

    if not ds_selected or not tech_selected or not eval_selected:
        typer.echo("No selections found. Use 'deid select <target> <items>' first.")
        raise typer.Exit(1)

    results_dir = loader.settings.result_path
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which stages need to run by checking existing result CSVs
    need_preprocess = set(ds_selected)  # datasets with no alignment
    need_techniques: dict[str, set[str]] = {ds: set(tech_selected) for ds in ds_selected}  # {dataset: set(techniques)}
    need_evals: dict[str, dict[str, set[str]]] = {}  # {dataset: {technique: set(evaluations)}}

    # Walk {technique}/{dataset}/*.csv hierarchy
    for tech_dir in results_dir.iterdir():
        if not tech_dir.is_dir() or tech_dir.name == "__pycache__":
            continue
        tech_name = tech_dir.name
        for ds_dir in tech_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            ds_name = ds_dir.name
            if ds_name not in set(ds_selected):
                continue
            need_techniques[ds_name].discard(tech_name)
            need_evals.setdefault(ds_name, {}).setdefault(tech_name, set(ds_selected)).discard(tech_name)
            for csv in ds_dir.glob("*.csv"):
                need_evals[ds_name][tech_name].discard(csv.stem)

    # Remove empty entries
    need_techniques = {ds: ts for ds, ts in need_techniques.items() if ts}
    # Validation runs automatically — never needs manual tracking
    need_techniques = {ds: ts - {"validation"} for ds, ts in need_techniques.items()}
    need_evals = {ds: {t: es for t, es in ev.items() if es and t != "validation"}
                  for ds, ev in need_evals.items() if ev}

    stages = []
    if need_techniques:
        stages.append(("preprocess", None))
    if need_techniques:
        stages.append(("techniques", need_techniques))
    if need_evals:
        stages.append(("evaluation", need_evals))

    if not stages:
        typer.echo("All stages already complete — nothing to run.")
        return

    typer.echo("Stages to run:")
    for name, detail in stages:
        typer.echo(f"  {name}")
    typer.echo()

    for name, detail in stages:
        if name == "preprocess":
            typer.echo("Running preprocess...")
            success = run_preprocess(loader)
            typer.echo("  OK" if success else "  FAILED")
        elif name == "techniques" and detail:
            # detail is {dataset: set(techniques)} → we need to swap to {technique: set(datasets)}
            tech_to_ds: dict[str, set[str]] = {}
            for ds, techs in detail.items():
                for t in techs:
                    tech_to_ds.setdefault(t, set()).add(ds)
            for t, ds_set in tech_to_ds.items():
                typer.echo(f"Running technique {t} on {', '.join(sorted(ds_set))}...")
                # Temporarily set the selections for this stage
                orig_ds = list(ds_selected)
                s.datasets.selected = sorted(ds_set)
                s.techniques.selected = [t]
                success = run_techniques(loader)
                s.datasets.selected = orig_ds
                typer.echo("  OK" if success else "  FAILED")
        elif name == "evaluation" and detail:
            # detail is {dataset: {technique: set(evaluations)}}
            for ds, tech_map in detail.items():
                for tech, evals in tech_map.items():
                    typer.echo(f"Running evaluations {', '.join(sorted(evals))} on {tech}/{ds}...")
                    # Temporarily set the selections for this stage
                    orig_ds = list(ds_selected)
                    orig_ev = list(eval_selected)
                    s.datasets.selected = [ds]
                    s.techniques.selected = [tech]
                    s.evaluation.selected = sorted(evals)
                    success = run_evaluations(loader)
                    s.datasets.selected = orig_ds
                    s.evaluation.selected = orig_ev
                    typer.echo("  OK" if success else "  FAILED")

    typer.echo("Resumed run complete.")


# -------------- --------------------------------------------------
# Top-level commands (no group)
# -------------- --------------------------------------------------

def cmd_show() -> None:
    """Show current configuration."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    s = loader.settings
    typer.echo(f"root_dir:       {s.root_dir}")
    typer.echo(f"result_dir:     {s.result_dir}")
    typer.echo(f"logs_dir:       {s.logs_dir}")
    typer.echo(f"datasets:       {s.datasets.selected}")
    typer.echo(f"techniques:     {s.techniques.selected}")
    typer.echo(f"evaluation:     {s.evaluation.selected}")
    typer.echo(f"environments:   {s.environments}")
    typer.echo(f"visualization:  {s.visualization.selections}")


def cmd_migrate(yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt.")) -> None:
    """Migrate config.ini + pipeline.yml → deid-config.yaml."""
    from deid.config.migrator import Migrator
    migrator = Migrator(Path("root_dir/deid-config.yaml"))
    mapping = migrator.migrate(dry_run=not yes)
    if not yes:
        typer.echo()
        resp = typer.prompt("Write deid-config.yaml? (yes/no)", default="no", show_default=False)
        if resp.lower() in {"y", "yes"}:
            migrator.migrate(dry_run=False)
            typer.echo("Migration complete.")
        else:
            typer.echo("Cancelled.")
    else:
        typer.echo("Migration complete (non-interactive).")


def cmd_explore(port: int = typer.Option(8501, "--port", "-p", help="Streamlit server port.")) -> None:
    """Launch the Streamlit result browser."""
    app_path = Path(__file__).resolve().parent.parent / "explore" / "app.py"
    if not app_path.exists():
        typer.echo(f"Error: Streamlit app not found at {app_path}")
        raise typer.Exit(1)
    typer.echo(f"Starting Streamlit at http://localhost:{port}")
    typer.echo("Press Ctrl+C to stop.")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), f"--server.port={port}", "--server.headless=true"],
        check=True,
    )


def cmd_migrate_structure() -> None:
    """Migrate legacy root_dir structure: add .gitkeep files, document new layout."""
    from deid.config.loader import ConfigLoader
    loader = ConfigLoader()
    workspace = loader.settings.root_path

    dirs = ["techniques", "evaluation", "environments", "datasets", "results", "logs"]
    for d in dirs:
        target = workspace / d
        if not target.exists():
            typer.echo(f"Creating {d}/")
            target.mkdir()
        gitkeep = target / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            typer.echo(f"  Added {d}/.gitkeep")

    # Add deprecation notice for root_dir if it exists at project level
    root_dir_legacy = Path("root_dir")
    if root_dir_legacy.exists():
        typer.echo()
        typer.echo("Note: root_dir/ still exists at project level. It is now the legacy path.")
        typer.echo("New workspaces should use a separate directory (e.g., ~/deid-workspace/)")
        typer.echo("with deid-config.yaml pointing to it via root_dir: .")
