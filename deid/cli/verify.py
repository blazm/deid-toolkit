"""`deid verify` — read-only diagnostics for dataset/technique preparation.

Checks (never writes anything):

  * config loads (deid-config.yaml)
  * per dataset: aligned image count; label CSV discovery + row coverage vs
    aligned images + Path resolution + gender/expression column availability;
    genuine & impostor pair file integrity (sampled)
  * per selected technique: output folder present (legacy ``deidentified/{tech}/{ds}``
    or dataset-root layout ``datasets/{Technique}/{ds}`` or flat ``{tech}/{ds}``),
    output count vs aligned count (shortfalls are WARN — technique failure logs
    are a supported condition)
  * SOTA stack info: ``root_dir/predictions/{ds}`` and ``root_dir/embeddings``
  * environment info: Python, torch / CUDA availability

Exit code: 1 if any FAIL, else 0.
"""
from __future__ import annotations

import csv
from pathlib import Path

import typer

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
_NON_DATASET_DIRS = {
    "aligned", "original", "labels", "pairs", "deidentified", "deid", "temp", "tmp",
}


def _c(code: str, text: str) -> str:
    try:
        import colorama
        colorama.just_fix_windows_console()
        colors = {"PASS": colorama.Fore.GREEN, "WARN": colorama.Fore.YELLOW,
                  "FAIL": colorama.Fore.RED, "INFO": colorama.Fore.CYAN}
        return f" {colors.get(code, '')}[{code:4s}]{colorama.Style.RESET_ALL} {text}"
    except Exception:
        return f" [{code:4s}] {text}"


def count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())


def find_labels_file(labels_dir: Path, ds: str):
    if not labels_dir.is_dir():
        return None
    exact = labels_dir / f"{ds}_labels.csv"
    if exact.exists():
        return exact
    cands = [p for p in sorted(labels_dir.glob(f"{ds}*.csv")) if p.stem != "README"]
    return cands[0] if cands else None


def check_pairs(pairs_dir: Path, ds: str, aligned_dir: Path, out) -> None:
    for kind in ("genuine", "impostor"):
        pf = pairs_dir / f"{ds}_{kind}_pairs.txt"
        if not pf.exists():
            out("WARN", f"{ds} pairs: {pf.name} not found")
            continue
        lines = [ln.strip() for ln in
                 pf.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if ln.strip()]
        if not lines:
            out("FAIL", f"{ds} pairs: {pf.name} is empty")
            continue
        bad = 0
        for ln in lines[:200]:
            parts = ln.split()
            if len(parts) != 4:
                bad += 1
                continue
            if not (aligned_dir / parts[1]).exists() or not (aligned_dir / parts[3]).exists():
                bad += 1
        if bad:
            out("WARN", f"{ds} pairs: {pf.name} — {len(lines)} lines, {bad} malformed/unresolvable (sampled 200)")
        else:
            out("PASS", f"{ds} pairs: {pf.name} — {len(lines)} lines ok (sampled 200)")


def check_labels(labels_path: Path, repo_root: Path, ds: str, n_img: int, out) -> None:
    lf = find_labels_file(labels_path, ds)
    if lf is None:
        out("WARN", f"{ds} labels: no label CSV found in {labels_path}")
        return
    try:
        with lf.open(newline="", encoding="utf-8-sig", errors="ignore") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as e:
        out("FAIL", f"{ds} labels: cannot parse {lf.name} ({e.__class__.__name__})")
        return
    if not rows:
        out("FAIL", f"{ds} labels: {lf.name} is empty")
        return
    cols = list(rows[0].keys())

    if "Path" in cols:
        missing = 0
        for r in rows:
            p = (r.get("Path") or "").strip()
            if not p:
                missing += 1
                continue
            cand = (repo_root / p) if not Path(p).exists() else Path(p)
            if not cand.exists():
                missing += 1
        cov = "labels==aligned" if len(rows) == n_img else f"labels {len(rows)} vs aligned {n_img}"
        gcol = next((c for c in ("Gender_code", "Gender") if c in cols)) if any(c in cols for c in ("Gender_code", "Gender")) else None
        ecol = next((c for c in ("Emotion_code", "emotional_value") if c in cols), None)
        gfilled = sum(1 for r in rows if (r.get(gcol) or "").strip() not in ("", "nan")) if gcol else 0
        extra = []
        if gcol:
            extra.append(f"{gcol} {gfilled}/{len(rows)}")
        if ecol:
            extra.append("expression col ok")
        if missing:
            out("FAIL", f"{ds} labels: {lf.name} — {missing}/{len(rows)} row paths unresolveble ({cov}" + (f"; {', '.join(extra)}" if extra else "") + ")")
        elif gcol and gfilled == 0:
            out("WARN", f"{ds} labels: {lf.name} — {gcol} column EMPTY ({cov}; {', '.join(extra)}). Fill before gender evaluation (see label_generation_csv/).")
        else:
            out("PASS", f"{ds} labels: {lf.name} — {cov}, all paths resolve" + (f" ({', '.join(extra)})" if extra else ""))
    else:
        out("WARN", f"{ds} labels: {lf.name} has no Path column (cols: {cols[:8]}…)")


def main(
    all_datasets: bool = typer.Option(False, "--all", help="Check every aligned dataset, not only the selected ones."),
    detail: bool = typer.Option(False, "--detail", help="Extra detail lines."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Show only non-PASS lines."),
) -> None:
    """Verify that datasets and techniques are properly prepared (read-only; never writes)."""
    from deid.config.loader import ConfigLoader

    fails = 0
    warns = 0

    def out(code: str, text: str) -> None:
        nonlocal fails, warns
        if not (quiet and code == "PASS"):
            typer.echo(_c(code, text))
        if code == "FAIL":
            fails += 1
        elif code == "WARN":
            warns += 1

    # 0 ── config
    try:
        loader = ConfigLoader()
        s = loader.settings
        out("PASS", f"config: {s.root_path}/deid-config.yaml loaded — datasets={s.datasets.selected}, techniques={s.techniques.selected}, {len(s.evaluation.selected)} evaluations selected")
    except Exception as e:
        out("FAIL", f"config: cannot load deid-config.yaml ({e.__class__.__name__}: {e})")
        raise typer.Exit(1)

    rroot = s.root_path
    datasets_path = s.datasets_path
    aligned_path = s.aligned_path
    if not rroot.is_dir():
        out("FAIL", f"root dir {rroot} does not exist — nothing to verify")
        raise typer.Exit(1)
    repo_root = rroot.parent

    # dataset roots (also used later for SOTA-layout technique discovery)
    dataset_dirs = (sorted(d for d in aligned_path.iterdir() if d.is_dir())
                    if aligned_path.is_dir() else [])
    tech_root_map = {}
    if datasets_path.is_dir():
        for d in datasets_path.iterdir():
            if d.is_dir() and d.name.lower() not in _NON_DATASET_DIRS:
                tech_root_map[d.name] = d

    # 1 ── datasets
    if all_datasets:
        ds_list = [d.name for d in dataset_dirs]
        if not ds_list:
            out("FAIL", f"no aligned datasets under {aligned_path}")
            raise typer.Exit(0)
    else:
        ds_list = list(s.datasets.selected)
        for d in ds_list:
            if not (aligned_path / d).is_dir():
                out("WARN", f"dataset {d}: selected but no aligned/ dir")

    for ds in ds_list:
        a_dir = aligned_path / ds
        n_img = count_images(a_dir)
        if n_img == 0:
            out("FAIL", f"dataset {ds}: no aligned images found ({a_dir})")
            continue
        out("PASS", f"dataset {ds}: {n_img} aligned images in {a_dir.relative_to(repo_root)}")
        check_labels(datasets_path / "labels", repo_root, ds, n_img, out)
        check_pairs(datasets_path / "pairs", ds, a_dir, out)

    # 2 ── technique outputs
    tkey = lambda name: name.lower().replace("_", "-")
    for t in [x for x in s.techniques.selected if x != "validation"]:
        matches = set()
        for base in (s.deid_path, datasets_path):
            for name in (t, tkey(t)):
                p = base / name
                if p.is_dir():
                    matches.add(p)
        for name, d in tech_root_map.items():
            if tkey(name) == tkey(t) or tkey(name).startswith(tkey(t)) or tkey(t).startswith(tkey(name)):
                matches.add(d)
        if not matches:
            out("WARN", f"technique {t}: no output folder found (checked {s.deid_path / t} and dataset-root dirs)")
            continue
        any_ds_hit = False
        for ds in ds_list:
            a_n = count_images(aligned_path / ds)
            for m in sorted(matches):
                dsdir = m / ds
                if dsdir.is_dir():
                    n = count_images(dsdir)
                    rel = m.name if m.parent == datasets_path else f"{m.parent.name}/{m.name}"
                    if a_n and n >= a_n:
                        out("PASS", f"technique {t} × {ds}: {n}/{a_n} outputs ({rel})")
                    elif n > 0:
                        out("WARN", f"technique {t} × {ds}: {n}/{a_n} outputs ({a_n - n} short — check the failure log of {t})")
                    else:
                        out("FAIL", f"technique {t} × {ds}: output dir {rel}/{ds} empty")
                    any_ds_hit = True
                    break
            if not any_ds_hit:
                flat = count_images(sorted(matches)[0])
                if flat:
                    out("INFO", f"technique {t}: {flat} flat images in {sorted(matches)[0].name} (no per-dataset subdirs)")
                    any_ds_hit = True
        if not any_ds_hit and ds_list:
            out("WARN", f"technique {t}: folder(s) {[m.name for m in sorted(matches)]} found but no per-dataset output for {ds_list}")

    # 3 ── SOTA stack (SwinFace / TransFace embeddings + predictions)
    emb = rroot / "embeddings"
    mods = []
    if emb.is_dir():
        mods = sorted(d.name for d in emb.iterdir() if d.is_dir())
    if mods:
        for mod in mods:
            for ds in ds_list:
                a_n = count_images(aligned_path / ds)
                e_dir = emb / mod / "aligned" / ds
                n_e = sum(1 for p in e_dir.iterdir() if p.suffix == ".npy") if e_dir.is_dir() else 0
                if not e_dir.is_dir() or n_e == 0:
                    out("WARN", f"SOTA embeddings: {mod} \u00d7 {ds} — MISSING ({e_dir.relative_to(repo_root)})")
                elif a_n and n_e >= a_n:
                    out("PASS", f"SOTA embeddings: {mod} \u00d7 {ds} — {n_e}/{a_n}")
                else:
                    out("WARN", f"SOTA embeddings: {mod} \u00d7 {ds} — partial ({n_e}/{a_n})")
            t_root = emb / mod / "datasets"
            if t_root.is_dir():
                techs = sorted(d.name for d in t_root.iterdir() if d.is_dir())
                out("INFO", f"SOTA embeddings technique dirs ({mod}): {techs}")
    preds = rroot / "predictions"
    if preds.is_dir():
        for ds in ds_list:
            pds = preds / ds
            if pds.is_dir():
                csvs = sorted(p.name for p in pds.glob("*.csv"))
                if csvs:
                    more = f" (+{len(csvs)-6} more)" if len(csvs) > 6 else ""
                    out("INFO", f"SOTA predictions: {ds} — {', '.join(csvs[:6])}{more}")
    if not mods and not preds.is_dir():
        out("INFO", "SOTA stack: no root_dir/embeddings or root_dir/predictions found yet")

    # 4 ── environment info
    import sys
    pyv = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    bits = [f"python {pyv}"]
    try:
        import torch
        bits.append(f"torch {torch.__version__} (cuda: {'yes' if torch.cuda.is_available() else 'NO'})")
    except Exception:
        bits.append("torch: not importable in this interpreter")
    out("INFO", "env: " + ", ".join(bits))

    typer.echo("")
    typer.echo(f"verify: {fails} FAIL, {warns} WARN — root_dir untouched (read-only check)")
    raise typer.Exit(1 if fails else 0)
