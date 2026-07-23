"""Pipeline orchestrator for DEID operations.

Replaces the old ``Techniques.do_run`` / ``Evaluations.do_run`` /
``Preprocessing.do_run`` chain with a single, cross-platform,
logging-enabled orchestrator.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

from deid.config.loader import ConfigLoader


def _resolve_builtin_script(package: str, name: str) -> Path | None:
    """Resolve a script path from a bundled package, or None if not found."""
    from importlib import resources

    pkg = resources.files(package) / f"{name}.py"
    # On editable install, pkg is already a real pathlib.Path
    if isinstance(pkg, Path):
        return pkg if pkg.exists() else None
    try:
        return Path(pkg.locate())
    except (ImportError, AttributeError, FileNotFoundError):
        return None

logger = logging.getLogger(__name__)

# -------------- ------  Platform detection  ----------------------------
def _get_shell_executable() -> str:
    if platform.system() == "Windows":
        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        for fallback in [git_bash, r"C:\Program Files (x86)\Git\bin\bash.exe"]:
            if os.path.exists(fallback):
                return fallback
        return "bash"  # hope it's on PATH (WSL / MSYS2)
    return "/bin/bash"


def _find_conda_sh() -> str:
    """Find conda.sh on the current platform."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "miniforge3", "etc", "profile.d", "conda.sh"),
        os.path.join(home, "anaconda3", "etc", "profile.d", "conda.sh"),
        os.path.join(home, "miniconda3", "etc", "profile.d", "conda.sh"),
    ]
    # Windows paths
    if platform.system() == "Windows":
        localappdata = os.environ.get("LOCALAPPDATA", "")
        for base in [home, localappdata]:
            for prefix in ["Miniforge3", "Anaconda3", "Miniconda3"]:
                p = os.path.join(base, prefix, "etc", "profile.d", "conda.sh")
                if os.path.exists(p):
                    return p
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(home, "miniforge3", "etc", "profile.d", "conda.sh")


# -------------- ------  Subprocess runner  ----------------------------
def _conda_env_exists(env_name: str) -> bool:
    """Check if a conda environment exists by name (directory check)."""
    home = os.path.expanduser("~")
    localappdata = os.environ.get("LOCALAPPDATA", "")
    env_dirs = [
        os.path.join(home, "miniforge3", "envs"),
        os.path.join(home, "anaconda3", "envs"),
        os.path.join(home, "miniconda3", "envs"),
        os.path.join(home, ".conda", "envs"),
        os.path.join(home, ".conda", "environments"),
    ]
    if platform.system() == "Windows":
        for base in [home, localappdata]:
            for prefix in ["Miniforge3", "Anaconda3", "Miniconda3"]:
                env_dirs.append(os.path.join(base, prefix, "envs"))
        env_dirs.append(os.path.join("C:", "Users", "conda", "envs"))
    for d in env_dirs:
        if os.path.isdir(os.path.join(d, env_name)):
            return True
    return False


def run_streamed(
    command: str,
    *,
    shell: bool = True,
    executable: Optional[str] = None,
    cwd: Optional[str] = None,
    env_extra: Optional[dict[str, str]] = None,
) -> int:
    """Run a command, streaming stdout/stderr to the console."""
    exe = executable or _get_shell_executable()

    # On Windows, passing a path with spaces as `executable` + `shell=True`
    # breaks because Windows splits on the first space.
    # Fix: use a list [executable, "-c", command] with shell=False.
    proc_env = os.environ.copy()
    if env_extra:
        proc_env.update(env_extra)

    if " " in exe and platform.system() == "Windows":
        proc = subprocess.Popen(
            [exe, "-c", command],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=proc_env,
        )
    else:
        proc = subprocess.Popen(
            command,
            shell=shell,
            executable=exe,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=proc_env,
        )

    try:
        import time

        last_time = time.time()
        cr_lines: list[str] = []  # buffered \r lines (progress bars)
        while True:
            raw = proc.stdout.readline()
            if not raw:
                if proc.poll() is not None:
                    break
                continue

            stripped = raw.rstrip("\r\n")
            if raw.rstrip("\n").endswith("\r"):
                # Carriage return line — tqdm progress bar; buffer it
                cr_lines.append(stripped.rstrip("\r"))
            else:
                # Flush any buffered progress bar before printing normal lines
                if cr_lines:
                    # Show the latest progress update within this flush window
                    print(cr_lines[-1])
                    cr_lines = []
                print(stripped)

            # Show buffered progress bar at least every 1 second
            now = time.time()
            if cr_lines and (now - last_time) >= 1.0:
                print(cr_lines[-1])
                cr_lines = []
                last_time = now
        # Flush remaining progress bar
        if cr_lines:
            print(cr_lines[-1])

        proc.wait()
        return proc.returncode
    finally:
        proc.stdout.close()


def run_technique(
    technique_name: str,
    aligned_dataset_path: Path,
    save_path: Path,
    loader: ConfigLoader,
) -> int:
    """Run a single technique script in its conda env."""
    settings = loader.settings
    envs = loader.load_environments()
    venv_name = envs.get(technique_name, technique_name)

    conda_sh = _find_conda_sh()
    if not os.path.exists(conda_sh):
        logger.error("conda.sh not found — cannot run %s", technique_name)
        return 1

    # Check user directory first (user-provided script), then fall back to built-in
    tech_path = settings.techniques_path / f"{technique_name}.py"
    if not tech_path.exists():
        builtin = _resolve_builtin_script("deid.techniques", technique_name)
        if builtin:
            tech_path = builtin
    if not tech_path.exists():
        logger.error("Technique script not found: %s", technique_name)
        return 1

    aligned_abs = str(aligned_dataset_path.resolve())
    save_abs = str(save_path.resolve())
    tech_abs = str(tech_path.resolve())

    # Build args (including any technique-specific args from pipeline.yml)
    args_map = loader.settings.techniques.args  # type: ignore[union-attr]
    extra_args = ""
    if args_map and technique_name in args_map:
        extra_args = f" {args_map[technique_name]}"

    # DeepPrivacy2 needs dataset file type detection
    filetype = ""
    if technique_name == "deepprivacy2":
        sample = next((aligned_dataset_path).iterdir(), None)
        if sample:
            ext = sample.suffix.lstrip(".")
            filetype = f" --dataset_filetype {ext} --dataset_newtype {ext}"

    # Fall back to current Python (sys.executable) if conda env doesn't exist
    this_python = sys.executable
    if _conda_env_exists(venv_name):
        py_cmd = f'source "{conda_sh}" && conda activate {venv_name} && python -u'
    else:
        logger.info("Conda env '%s' not found — using current Python", venv_name)
        py_cmd = f'"{this_python}" -u'

    cmd = (
        f'{py_cmd} "{tech_abs}" "{aligned_abs}" "{save_abs}"{extra_args}{filetype}'
    )

    logger.info("Running: %s", cmd)
    return run_streamed(cmd)


def run_evaluation(
    evaluation_name: str,
    aligned_dataset_path: Path,
    deid_dataset_path: Path,
    dataset_name: str,
    technique_name: str,
    impostor_pairs: Path,
    genuine_pairs: Path,
    save_path: Path,
    loader: ConfigLoader,
) -> int:
    """Run a single evaluation script in its conda env."""
    settings = loader.settings
    envs = loader.load_environments()
    venv_name = envs.get(evaluation_name, evaluation_name)

    conda_sh = _find_conda_sh()
    # Check user directory first, then fall back to built-in
    eval_script = settings.evaluation_path / f"{evaluation_name}.py"
    if not eval_script.exists():
        builtin = _resolve_builtin_script("deid.evaluation", evaluation_name)
        if builtin:
            eval_script = builtin
    if not eval_script.exists():
        logger.error("Evaluation script not found: %s", evaluation_name)
        return 1

    # Resolve eval_package_dir for path-relative imports in eval scripts
    eval_package_dir = eval_script.parent

    # Fall back to current Python (sys.executable) if conda env doesn't exist
    this_python = sys.executable
    if _conda_env_exists(venv_name):
        py_cmd = f'source "{conda_sh}" && conda activate {venv_name} && python -u'
    else:
        logger.info("Conda env '%s' not found — using current Python", venv_name)
        py_cmd = f'"{this_python}" -u'

    cmd = (
        f'{py_cmd} "{str(eval_script.resolve())}" '
        f'"{str(aligned_dataset_path.resolve())}" '
        f'"{str(deid_dataset_path.resolve())}" '
        f'--dataset_name {dataset_name} '
        f'--technique_name {technique_name} '
        f'--impostor_pairs_filepath "{str(impostor_pairs.resolve())}" '
        f'--genuine_pairs_filepath "{str(genuine_pairs.resolve())}" '
        f'--save_path "{str(save_path.resolve())}" '
        f'--root_dir "{str(settings.root_path.resolve())}" '
        f'--eval_package_dir "{str(eval_package_dir.resolve())}"'
    )

    logger.info("Running evaluation: %s", evaluation_name)
    return run_streamed(cmd, env_extra={"PYTHONPATH": str(eval_package_dir.resolve())})


# -------------- ------  Pipeline steps  ----------------------------
def _check_label_extraction(selected_datasets: list[str], loader: ConfigLoader) -> None:
    """Warn if any selected dataset lacks a corresponding label extraction script.

    Label extraction scripts live in ``label_generation_csv/`` and are named
    ``{dataset}_labels.py`` (case-insensitive match).
    """
    label_script_dir = Path(__file__).resolve().parent.parent.parent / "label_generation_csv"
    if not label_script_dir.is_dir():
        return  # No label extraction directory exists

    missing = []
    for ds in selected_datasets:
        found = False
        for script in label_script_dir.iterdir():
            if script.suffix == ".py":
                script_ds = script.stem.lower().replace("_labels", "")
                if ds.lower() in script_ds or script_ds in ds.lower():
                    found = True
                    break
        if not found:
            missing.append(ds)

    if missing:
        logger.warning(
            "The following datasets have no label extraction script in label_generation_csv/: %s\n"
            "   Labels for evaluation (pairs, identities) will be missing.\n"
            "   Create a script in label_generation_csv/{ds}_labels.py to generate labels.",
            ", ".join(missing),
        )


def run_preprocess(loader: ConfigLoader) -> bool:
    """Run preprocessing: alignment + pair generation."""
    from deid.utils import align_face_mtcnn as align_mtcnn
    from deid.utils import generate_img_pairs_all as gen_pairs

    settings = loader.settings
    selected = loader.settings.datasets.selected  # type: ignore[union-attr]
    if not selected:
        logger.warning("No datasets selected for preprocessing")
        return False

    # Check for missing label extraction scripts
    _check_label_extraction(selected, loader)

    success = True
    for ds_name in selected:
        aligned = settings.aligned_path / ds_name
        original = settings.original_path / ds_name / "img"
        # Prefer pre-aligned images; fall back to original for alignment
        if aligned.exists():
            logger.info("Using pre-aligned images for %s", ds_name)
        elif original.exists():
            logger.info("Aligning %s from original images", ds_name)
        else:
            logger.warning("Neither aligned nor original dataset found: %s — skipping", ds_name)
            continue

        aligned.mkdir(parents=True, exist_ok=True)
        if original.exists():
            try:
                align_mtcnn.mp_main(
                    dataset_path=str(original),
                    dataset_save_path=str(aligned),
                    dataset_name=ds_name,
                )
            except Exception as exc:
                logger.error("Alignment failed for %s: %s", ds_name, exc)
                success = False

    # Generate image pairs
    pairs_dir = settings.datasets_path / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    try:
        gen_pairs.main(selected, str(settings.datasets_path / "labels"), str(pairs_dir))
    except Exception as exc:
        logger.error("Pair generation failed: %s", exc)
        success = False

    return success


def run_techniques(loader: ConfigLoader) -> bool:
    """Run selected techniques on selected datasets."""
    settings = loader.settings
    ds_selected = loader.settings.datasets.selected  # type: ignore[union-attr]
    tech_selected = loader.settings.techniques.selected  # type: ignore[union-attr]
    if not ds_selected or not tech_selected:
        logger.warning("No datasets or techniques selected")
        return False

    success = True
    for tech in tech_selected:
        if tech == "validation":
            continue  # Validation is hardcoded — no technique script or conda env needed
        for ds in ds_selected:
            aligned = settings.aligned_path / ds
            save_path = settings.deid_path / tech / ds
            save_path.mkdir(parents=True, exist_ok=True)

            result = run_technique(tech, aligned, save_path, loader)
            if result != 0:
                logger.error("Technique %s failed on %s (exit code %d)", tech, ds, result)
                success = False

    return success


def run_evaluations(loader: ConfigLoader) -> bool:
    """Run selected evaluations on selected datasets+techniques."""
    settings = loader.settings
    ds_selected = loader.settings.datasets.selected  # type: ignore[union-attr]
    tech_selected = loader.settings.techniques.selected  # type: ignore[union-attr]
    eval_selected = loader.settings.evaluation.selected  # type: ignore[union-attr]
    if not ds_selected or not eval_selected:
        logger.warning("No datasets or evaluations selected")
        return False
    # tech_selected may be empty when only validation is needed
    if tech_selected and "validation" not in tech_selected:
        tech_selected = [t for t in tech_selected if t != "validation"]
        if not tech_selected:
            logger.warning("Only 'validation' technique selected — use 'deid run validation' or 'deid run all'")
            return False

    result_dir = settings.result_path
    result_dir.mkdir(parents=True, exist_ok=True)

    pairs_dir = settings.datasets_path / "pairs"
    success = True
    for ev in eval_selected:
        for tech in tech_selected:
            for ds in ds_selected:
                aligned = settings.aligned_path / ds
                deid = settings.deid_path / tech / ds
                if not deid.exists():
                    logger.warning("No deid output for %s/%s — skipping", tech, ds)
                    continue

                impostor = pairs_dir / f"{ds}_impostor_pairs.txt"
                genuine = pairs_dir / f"{ds}_genuine_pairs.txt"
                if not impostor.exists() or not genuine.exists():
                    logger.warning("Pair files missing for %s — skipping", ds)
                    continue

                # New hierarchy: results/{tech}/{ds}/{eval}.csv
                tech_ds_dir = result_dir / tech / ds
                tech_ds_dir.mkdir(parents=True, exist_ok=True)
                save = tech_ds_dir / f"{ev}.csv"
                result = run_evaluation(
                    ev, aligned, deid, ds, tech, impostor, genuine, save, loader
                )
                if result != 0:
                    logger.error("Evaluation %s failed for %s/%s", ev, ds, tech)
                    success = False

    # Validation: always run aligned images as reference.
    for ev in eval_selected:
        for ds in ds_selected:
            aligned = settings.aligned_path / ds
            impostor = pairs_dir / f"{ds}_impostor_pairs.txt"
            genuine = pairs_dir / f"{ds}_genuine_pairs.txt"
            if not impostor.exists() or not genuine.exists():
                logger.warning("Pair files missing for %s — skipping validation", ds)
                continue
            val_dir = result_dir / "validation" / ds
            val_dir.mkdir(parents=True, exist_ok=True)
            save = val_dir / f"{ev}.csv"
            result = run_evaluation(
                ev, aligned, aligned, ds, "validation", impostor, genuine, save, loader
            )
            if result != 0:
                logger.error("Validation eval %s failed for %s", ev, ds)
                success = False

    return success


def _write_manifest(result_dir: Path, loader: ConfigLoader) -> None:
    """Write a manifest.json with pipeline run metadata."""
    settings = loader.settings
    manifest = {
        "version": 1,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "root_dir": settings.root_dir,
        "result_dir": settings.result_dir,
        "logs_dir": settings.logs_dir,
        "datasets": list(settings.datasets.selected),
        "techniques": list(settings.techniques.selected),
        "evaluation": list(settings.evaluation.selected),
        "technique_args": dict(settings.techniques.args) if settings.techniques.args else {},
        "dataset_renames": dict(settings.dataset_renames) if settings.dataset_renames else {},
        "technique_renames": dict(settings.technique_renames) if settings.technique_renames else {},
        "evaluation_renames": dict(settings.evaluation_renames) if settings.evaluation_renames else {},
    }
    manifest_path = result_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_all(loader: ConfigLoader) -> bool:
    """Execute the full pipeline: preprocess -> techniques -> evaluations."""
    logger.info("Starting full pipeline")
    results = []
    results.append(("preprocess", run_preprocess(loader)))
    results.append(("techniques", run_techniques(loader)))
    results.append(("evaluations", run_evaluations(loader)))

    for name, ok in results:
        status = "OK" if ok else "FAILED"
        logger.info("  %s: %s", name, status)

    # Write manifest with run metadata
    _write_manifest(loader.settings.result_path, loader)

    # Generate PDF reports from results (non-critical — requires matplotlib)
    try:
        from deid.reports.pdf_export import export_results_to_pdf, generate_summary_report
        result_dir = loader.settings.result_path
        pdf_reports = export_results_to_pdf(result_dir)
        summary = generate_summary_report(result_dir)
        logger.info("Generated %d PDF reports, summary at %s", len(pdf_reports), summary)
    except ImportError:
        logger.info("PDF export skipped — install reports extra: pip install -e \".[reports]\"")
    except Exception as exc:
        logger.warning("PDF export failed (non-critical): %s", exc)

    return all(ok for _, ok in results)

