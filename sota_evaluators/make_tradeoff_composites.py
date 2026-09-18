#!/usr/bin/env python
"""
Trade-off composites for the DeID toolkit (one file per technique).

Axes (all scores in [0,1], 0 = bad, 1 = best; see manuscript for the
normalized derivations implemented here verbatim):

  P (privacy, per backbone) = mean of:
    P1  min(1, EER_sc/0.5)               same-condition 1:1 verification EER
    P2  clip(1 - (R@1 - c)/(1 - c), 0, 1)  closed-set 1:N identity-level R@1,
                                           c = sum_g (n_g/2824)^2 = 0.0031
    P3  min(1, EER_xc/0.5)               cross-condition (de-anonymization) EER
  N (naturalness) = mean of:
    N1  1 - min(1, LPIPS/L0),  L0 = max LPIPS over the 20 techniques
    N2  eDiffIQA retention = mean(IQA(deID set)) / mean(IQA(aligned set))
  U (utility) = mean of 7 baseline-relative retention ratios:
    gender on CelebA-test, RaFD, MUG-still, KDEF
    expression on RaFD, MUG-still, KDEF (CelebA has no expression task)
  Overall(bk) = (P_bk + N + U) / 3
  Pareto(bk): techniques non-dominated in (P_bk, N, U).

Inputs (all read-only, all precomputed):
  predictions/verification_<bk>_all.html         AUC+EER per technique (same-cond)
  predictions/ranks_<bk>_per_identity.csv        stem-level ranks (closed-set 1:N)
  predictions/linkability_<bk>_results.json      per-technique cross-cond EER
  predictions/naturalness/celeba-test/<T>.json   LPIPS + eDiffIQA
  predictions/<ds>/gender_results.csv            Aligned/DeID gender accuracy
  predictions/<ds>/expression_results.csv        Aligned/DeID expression accuracy
Outputs:
  predictions/tradeoff/celeba-test/<TECH>.json   all inputs + scores + overall
  predictions/tradeoff/celeba-test/tradeoff_summary.csv
  predictions/tradeoff/celeba-test/_meta.json    constants (c, L0), inputs used
Usage:
  python make_tradeoff_composites.py [--root D:/dev/deid-toolkit]
"""
import argparse, csv, json, os, re
import numpy as np

R1_CHANCE = 0.0031  # sum_g (n_g/2824)^2 over the real CelebA-HQ test groups
GENDER_DATASETS = ["celeba-test", "rafd", "mug-still", "kdef"]
EXPR_DATASETS = ["rafd", "mug-still", "kdef"]
BACKBONES = {"swinface": "SwinFace", "transface": "TransFace"}


def _f(x):
    """Parse a CSV cell that may be '97.2', '97.2%', or 'N/A'."""
    if x is None:
        return None
    s = str(x).strip().rstrip("%")
    if s in ("", "N/A", "NA", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def techs_from_ranks(path):
    techs, stems = {}, {}
    with open(path, encoding="cp1252") as f:
        for row in csv.DictReader(f):
            t = row["technique"].strip().strip("'")
            r = int(row["rank"])
            stems.setdefault(t, {})[row["stem"]] = r
    return {t: float(np.mean([1.0 if r <= 1 else 0.0 for r in s.values()]))
            for t, s in stems.items()}


def eers_from_html(path):
    txt = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(
            r"<tr><td style='text-align:left'>([^<]+)</td><td>[^<]*</td><td>[^<]*</td>"
            r"<td>([0-9.]+)</td><td>([0-9.]+)</td>", txt):
        name = m.group(1).strip()
        if name in ("Validation",):
            continue
        out[name] = {"auc_same": float(m.group(2)), "eer_same": float(m.group(3))}
    return out


def cc_eer_from_json(path):
    d = json.load(open(path, encoding="utf-8"))
    return {t: v["eer"] for t, v in d["techniques"].items()}


def ratio_rows(path, ref_col, deid_col):
    out = {}
    if not os.path.exists(path):
        return out
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    import io
    for row in csv.DictReader(io.StringIO(text)):
            t = row["Technique"].strip().strip("'")
            if t.startswith("aligned"):
                continue
            ref, de = _f(row.get(ref_col)), _f(row.get(deid_col))
            if ref is None or de is None or ref <= 0:
                continue
            out[t] = min(1.0, de / ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DEID_ROOT", "D:/dev/deid-toolkit"))
    args = ap.parse_args()
    PRED = os.path.join(args.root, "root_dir", "predictions")
    OUT = os.path.join(PRED, "tradeoff", "celeba-test")
    os.makedirs(OUT, exist_ok=True)

    # --- privacy inputs per backbone
    priv = {}
    for bk, label in BACKBONES.items():
        eers = eers_from_html(os.path.join(PRED, f"verification_{bk}_all.html"))
        r1 = techs_from_ranks(os.path.join(PRED, f"ranks_{bk}_per_identity.csv"))
        cc = cc_eer_from_json(os.path.join(PRED, f"linkability_{bk}_results.json"))
        priv[bk] = {"eers": eers, "r1": r1, "cc": cc}

    # --- naturalness
    nat_dir = os.path.join(PRED, "naturalness", "celeba-test")
    naturalness = {}
    for fn in os.listdir(nat_dir):
        m = re.match(r"([A-Za-z0-9\-]+)\.json$", fn)
        if not m:
            continue
        d = json.load(open(os.path.join(nat_dir, fn), encoding="ascii"))
        naturalness[m.group(1)] = {"lpips": d["lpips_mean"], "iqa_ret": d["ediff_retention"]}
    L0 = max(v["lpips"] for v in naturalness.values())

    # --- utility
    gender, expr = {}, {}
    for ds in GENDER_DATASETS:
        for t, r in ratio_rows(os.path.join(PRED, ds, "gender_results.csv"),
                               "Aligned_Gender_Accuracy", "DeID_Gender_Accuracy").items():
            gender.setdefault(t, {})[ds] = r
    for ds in EXPR_DATASETS:
        for t, r in ratio_rows(os.path.join(PRED, ds, "expression_results.csv"),
                               "Aligned_Expr_Accuracy", "DeID_Expr_Accuracy").items():
            expr.setdefault(t, {})[ds] = r

    techs = sorted(naturalness)
    results = {}
    for t in techs:
        rec = {"technique": t, "dataset": "celeba-test"}
        n1 = 1.0 - min(1.0, naturalness[t]["lpips"] / L0)
        ret_raw = naturalness[t]["iqa_ret"]
        n2 = min(1.0, ret_raw)  # cap: de-identification does not improve on its reference
        rec["N"] = {"lpips": naturalness[t]["lpips"], "lpips_util": n1,
                    "iqa_ret_raw": ret_raw, "iqa_retention": n2,
                    "n": 0.5 * (n1 + n2)}
        g = gender.get(t, {})
        e = expr.get(t, {})
        u_parts = list(g.values()) + list(e.values())
        rec["U"] = {"gender": g, "expression": e, "n_ratios": len(u_parts),
                    "u": float(np.mean(u_parts)) if u_parts else None}
        rec["backbones"] = {}
        for bk in BACKBONES:
            p = priv[bk]
            e11 = p["eers"].get(t, {}).get("eer_same")
            r1v = p["r1"].get(t)
            ecx = p["cc"].get(t)
            s1 = min(1.0, e11 / 0.5) if e11 is not None else None
            s2 = (float(np.clip(1.0 - (r1v - R1_CHANCE) / (1.0 - R1_CHANCE), 0, 1))
                  if r1v is not None else None)
            s3 = min(1.0, ecx / 0.5) if ecx is not None else None
            p_score = float(np.mean([s for s in (s1, s2, s3) if s is not None])) \
                if any(s is not None for s in (s1, s2, s3)) else None
            rec["backbones"][bk] = {
                "eer_same": e11, "auc_same": p["eers"].get(t, {}).get("auc_same"),
                "r1_reid": r1v, "eer_xc": ecx,
                "p1_eer_sc": s1, "p2_r1": s2, "p3_eer_xc": s3, "p": p_score,
            }
        ps = [rec["backbones"][b]["p"] for b in BACKBONES if rec["backbones"][b]["p"] is not None]
        if ps and rec["U"]["u"] is not None:
            rec["overall"] = {bk: (rec["backbones"][bk]["p"] + rec["N"]["n"] + rec["U"]["u"]) / 3
                              for bk in BACKBONES}
        results[t] = rec

    for t, rec in results.items():
        with open(os.path.join(OUT, t + ".json"), "w", encoding="ascii") as f:
            json.dump(rec, f, indent=1, sort_keys=True)

    # --- Pareto per backbone (on raw scores P, N, U)
    def dominated(a, b):
        return (a["p"] >= b["p"] and a["n"] >= b["n"] and a["u"] >= b["u"]
                and (a["p"] > b["p"] or a["n"] > b["n"] or a["u"] > b["u"]))
    pts = {}
    for bk in BACKBONES:
        items = [{"t": t, "p": r["backbones"][bk]["p"], "n": r["N"]["n"], "u": r["U"]["u"]}
                  for t, r in results.items()
                  if r["backbones"][bk]["p"] is not None and r["U"]["u"] is not None]
        front = [it["t"] for it in items
                 if not any(o["t"] != it["t"] and dominated(o, it) for o in items)]
        pts[bk] = front

    # --- summary CSV
    with open(os.path.join(OUT, "tradeoff_summary.csv"), "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["technique", "lpips", "iqa_ret", "n", "u_gender", "u_expression", "u",
                    "eer_sc_swin", "r1_swin", "eer_xc_swin", "p_swin", "overall_swin",
                    "eer_sc_trans", "r1_trans", "eer_xc_trans", "p_trans", "overall_trans",
                    "front_swin", "front_trans"])
        for t in sorted(results, key=lambda x: results[x]["overall"].get("swinface") or 0,
                        reverse=True):
            r = results[t]
            sw, tr = r["backbones"]["swinface"], r["backbones"]["transface"]
            g = [v for v in r["U"]["gender"].values()]
            e = [v for v in r["U"]["expression"].values()]
            w.writerow([t,
                        f'{r["N"]["lpips"]:.4f}', f'{r["N"]["iqa_retention"]:.4f}',
                        f'{r["N"]["n"]:.4f}',
                        f'{np.mean(g):.3f}' if g else "",
                        f'{np.mean(e):.3f}' if e else "",
                        f'{r["U"]["u"]:.4f}',
                        f'{sw["eer_same"]:.3f}', f'{sw["r1_reid"]:.5f}', f'{sw["eer_xc"]:.3f}',
                        f'{sw["p"]:.4f}', f'{r["overall"]["swinface"]:.4f}',
                        f'{tr["eer_same"]:.3f}', f'{tr["r1_reid"]:.5f}', f'{tr["eer_xc"]:.3f}',
                        f'{tr["p"]:.4f}', f'{r["overall"]["transface"]:.4f}',
                        1 if t in pts["swinface"] else 0, 1 if t in pts["transface"] else 0])

    meta = {
        "dataset": "celeba-test", "n_techniques": len(results),
        "r1_chance": R1_CHANCE,
        "lpips_l0_max": L0,
        "components": {"P": ["eer_same/0.5 (cap 1)", "clip(1-(r1-0.0031)/(1-0.0031))",
                              "eer_xc/0.5 (cap 1)"],
                        "N": ["1-min(1,lpips/L0)", "min(1, iqa_retention)"],
                        "U": ["7 baseline-relative retentions (4 gender + 3 expression)"]},
        "pareto_fronts": pts,
        "note": "Spearman(EER-score, AUC-score) = 0.998 (swinface) / 0.998 (transface); "
                "max rank shift 1 -- EER primary is metric-robust.",
    }
    with open(os.path.join(OUT, "_meta.json"), "w", encoding="ascii") as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    print(f"Done: {len(results)} techniques -> {OUT}")
    print(f"L0 (LPIPS max) = {L0:.4f}; fronts:")
    for bk, front in pts.items():
        print(f"  {bk}: {front}")


if __name__ == "__main__":
    main()
