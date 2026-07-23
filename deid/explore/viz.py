"""Plotting helpers for the Streamlit explore app."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_score_distribution(
    df: "pd.DataFrame",
    score_col: str,
    technique: str,
    dataset: str,
    eval_name: str,
) -> plt.Figure:
    """Plot score distribution for a single technique/dataset/evaluation."""
    import pandas as pd  # type: ignore[import-not-found]

    scores = df[score_col].dropna()
    if scores.empty:
        return plt.Figure()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(scores, kde=True, ax=ax)
    ax.set_title(f"{eval_name} — {dataset} / {technique}")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_metric_table(df: "pd.DataFrame") -> plt.Figure:
    """Plot a heatmap-style table of metrics."""
    import pandas as pd

    # Try to pivot into a metrics x techniques matrix
    if "metric" in df.columns and "score" in df.columns:
        pivot = df.pivot_table(index="dataset", columns="metric", values="score", aggfunc="mean")
    elif len(df.columns) > 1:
        pivot = df.iloc[:, 1:].mean()
    else:
        return plt.Figure()

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot.index) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Average Metric Scores")
    fig.tight_layout()
    return fig


def _find_score_and_label_cols(df: "pd.DataFrame") -> tuple[str | None, str | None]:
    """Find a continuous score column and a binary ground-truth column in the DataFrame.

    Returns (score_col, label_col) or (None, None).
    """
    import pandas as pd

    score_candidates = [c for c in df.columns if c.lower() in ("score", "sim", "simscore", "cosine", "similarity", "similarity_score", "cosine_similarity")]
    if not score_candidates:
        score_candidates = [c for c in df.columns if "score" in c.lower() or "sim" in c.lower()]
    score_col = score_candidates[0] if score_candidates else None

    label_candidates = [c for c in df.columns if "ground_truth" in c.lower() or "label" in c.lower() or "gt" in c.lower()]
    label_col = label_candidates[0] if label_candidates else None

    return score_col, label_col


def plot_roc_curve(
    df: "pd.DataFrame",
    technique: str,
    dataset: str,
    eval_name: str,
) -> tuple[plt.Figure, float | None]:
    """Plot an ROC curve from verification-based evaluation results.

    Requires a binary ground-truth column and a continuous score column.
    Returns (figure, auc_score) or (empty_figure, None).
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc  # type: ignore

    score_col, label_col = _find_score_and_label_cols(df)
    if score_col is None or label_col is None:
        return plt.Figure(), None

    labels = df[label_col].dropna().astype(int).values
    scores = df[score_col].dropna().values
    valid = labels >= 0  # filter out any unlabeled rows

    if valid.sum() < 2:
        return plt.Figure(), None

    labels = labels[valid]
    scores = scores[valid]

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {eval_name} | {dataset} / {technique}")
    ax.legend(loc="lower right")

    # Show best threshold via Youden's J
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    ax.scatter(
        [fpr[best_idx]], [tpr[best_idx]],
        color="green", zorder=5,
        label=f"Best threshold: {thresholds[best_idx]:.3f} (J={j_scores[best_idx]:.3f})",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, roc_auc


def plot_pr_curve(
    df: "pd.DataFrame",
    technique: str,
    dataset: str,
    eval_name: str,
) -> tuple[plt.Figure, float | None]:
    """Plot a Precision-Recall curve from verification-based evaluation results.

    Returns (figure, average_precision) or (empty_figure, None).
    """
    import numpy as np
    from sklearn.metrics import precision_recall_curve, average_precision_score  # type: ignore

    score_col, label_col = _find_score_and_label_cols(df)
    if score_col is None or label_col is None:
        return plt.Figure(), None

    labels = df[label_col].dropna().astype(int).values
    scores = df[score_col].dropna().values
    valid = labels >= 0

    if valid.sum() < 2:
        return plt.Figure(), None

    labels = labels[valid]
    scores = scores[valid]

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="blue", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {eval_name} | {dataset} / {technique}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig, ap


def plot_cmc_curve(
    df: "pd.DataFrame",
    technique: str,
    dataset: str,
    eval_name: str,
) -> plt.Figure:
    """Plot a Cumulative Matching Characteristic (CMC) curve from identification results.

    Requires columns: rank, true_identity.
    Shows fraction of probes whose true identity appears in top-k gallery matches.
    """
    import pandas as pd

    rank_col = "rank" if "rank" in df.columns else None
    if not rank_col:
        return plt.Figure()

    ranks = df[rank_col].dropna().astype(int)
    if ranks.empty:
        return plt.Figure()

    n = len(ranks)
    # CMC: fraction of probes with rank <= k
    max_rank = min(ranks.max(), 50)  # cap at 50
    cum_rates = []
    for k in range(1, max_rank + 1):
        cum_rates.append((ranks <= k).sum() / n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, max_rank + 1), cum_rates, marker="o", markersize=4, lw=2, color="darkorange")
    # Add rank@1, rank@5, rank@10 highlights
    for k in [1, 5, 10]:
        if k <= max_rank:
            ax.scatter([k], [cum_rates[k - 1]], color="green", s=60, zorder=5)
            ax.annotate(
                f"{100*cum_rates[k-1]:.0f}%",
                (k, cum_rates[k - 1]),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=9,
                color="green",
            )
    ax.set_xlim([1, max_rank])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Rank (Top-k)")
    ax.set_ylabel("Cumulative Matching Rate")
    ax.set_title(f"CMC Curve — {eval_name} | {dataset} / {technique}")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max_rank + 1, min(max_rank // 5, 5)))
    fig.tight_layout()
    return fig


def plot_cmc_multi(
    dfs: dict[str, "pd.DataFrame"],  # technique -> DataFrame
    dataset: str,
    eval_name: str,
    max_rank: int = 50,
) -> plt.Figure:
    """Plot multiple CMC curves on one axis for comparing techniques.

    dfs: {technique_name: identification_csv_dataframe}
    """
    import pandas as pd

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("husl", len(dfs))

    for (tech, df), color in zip(dfs.items(), colors):
        ranks = df["rank"].dropna().astype(int)
        if ranks.empty:
            continue
        m = min(ranks.max(), max_rank)
        cum = [(ranks <= k).sum() / len(ranks) for k in range(1, m + 1)]
        ax.plot(range(1, m + 1), cum, marker="o", markersize=4, lw=2,
                color=color, label=f"{tech} (rank@1={100*cum[0]:.0f}%)")

    ax.set_xlim([1, max_rank])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Rank (Top-k)")
    ax.set_ylabel("Cumulative Matching Rate")
    ax.set_title(f"CMC Comparison — {eval_name} | {dataset}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max_rank + 1, min(max_rank // 5, 5)))
    fig.tight_layout()
    return fig


def plot_embedding_clustering(
    embeddings: pd.DataFrame,  # columns: x, y, identity, source (original/deid)
    technique: str,
    dataset: str,
) -> plt.Figure:
    """Scatter plot of 2D projected embeddings colored by identity.

    embeddings: DataFrame with columns x, y, identity, source
    Compares original vs de-identified embedding space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col_idx, source in enumerate(["original", "deid"]):
        ax = axes[col_idx]
        subset = embeddings[embeddings["source"] == source]
        identities = subset["identity"].unique()

        palette = sns.color_palette("husl", min(len(identities), 50))
        for i, ident in enumerate(sorted(identities)):
            mask = subset["identity"] == ident
            color = palette[i % len(palette)]
            ax.scatter(
                subset.loc[mask, "x"], subset.loc[mask, "y"],
                color=color, alpha=0.6, s=20, label=ident,
            )

        ax.set_title(f"{source.title()} — {dataset} / {technique}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(fontsize=7, loc="best", ncol=3)
        if len(identities) > 30:
            ax.get_legend().remove()
            ax.text(
                0.01, 0.99, f"{len(identities)} identities",
                transform=ax.transAxes, va="top", fontsize=9,
            )

    fig.suptitle(f"Embedding Space — Original vs {technique.title()}", fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================
# Ported from legacy/visualization/
# ============================================================


def plot_confusion_matrix(
    df_results: pd.DataFrame,  # evaluation results CSV with columns: img, aligned_predictions, deidentified_predictions (or Emotion_code)
    df_labels: pd.DataFrame,  # labels CSV with columns: Name, Emotion_code
    eval_name: str,
    technique: str,
    dataset: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot confusion matrix from emotion recognition results.

    df_results: evaluation CSV (deidentified_predictions or aligned_predictions columns)
    df_labels: dataset labels CSV (Emotion_code column)
    Returns (figure, formatted DataFrame) for use in tables.
    """
    # Map emotion codes to labels
    labels_map = {
        0: "Neutral", 1: "Anger", 2: "Scream", 3: "Contempt",
        4: "Disgust", 5: "Fear", 6: "Happy", 7: "Sadness", 8: "Surprise",
    }
    expression_labels = list(labels_map.values())
    cmap = plt.cm.jet
    cmap_with_grey = matplotlib.colors.ListedColormap(["lightgrey"] + [cmap(i) for i in range(cmap.N)])

    # Get predictions — prefer deidentified, fall back to aligned
    pred_col = "deidentified_predictions" if "deidentified_predictions" in df_results.columns else None
    if pred_col is None:
        pred_col = "aligned_predictions" if "aligned_predictions" in df_results.columns else None
    if pred_col is None:
        return plt.Figure(), pd.DataFrame()

    true_col = "Emotion_code" if "Emotion_code" in df_labels.columns else None

    # Build confusion matrix
    matrix = np.zeros((len(expression_labels), len(expression_labels)))
    n_correct = 0
    n_total = 0

    for _, row in df_results.iterrows():
        img_name = row["img"] if "img" in df_results.columns else row.index[0]
        pred_val = row[pred_col]

        # Get ground truth from labels
        true_val = None
        if true_col:
            label_row = df_labels[df_labels["Name"] == img_name]
            if not label_row.empty:
                true_val = int(label_row.iloc[0][true_col])

        if true_val is None or not np.isfinite(pred_val):
            continue

        # Map pred_val to label name if possible
        pred_label = int(pred_val) if np.isfinite(pred_val) else pred_val
        if true_val in labels_map and pred_label in labels_map:
            x = expression_labels.index(labels_map[pred_label])
            y = expression_labels.index(labels_map[true_val])
            matrix[x, y] += 1
            if x == y:
                n_correct += 1
        n_total += 1

    # Compute per-emotion stats
    stats_rows = []
    for i, label in enumerate(expression_labels):
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        stats_rows.append({
            "label": label,
            "accuracy": f"{acc:.2%}",
            "count": int(tp + fn),
            "correct": int(tp),
        })

    stats_df = pd.DataFrame(stats_rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap=cmap_with_grey, interpolation="nearest")
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            val = matrix[x, y]
            color = "black" if np.isfinite(val) and matrix[x, y] > 0 else "grey"
            ax.text(y, x, str(int(val)), ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(len(expression_labels)))
    ax.set_xticklabels(expression_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(expression_labels)))
    ax.set_yticklabels(expression_labels, fontsize=7)
    ax.set_title(f"Confusion Matrix — {eval_name} | {dataset} / {technique}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    return fig, stats_df


def plot_distance_histogram(
    df: pd.DataFrame,
    technique: str,
    dataset: str,
    eval_name: str,
    score_col: str = "cossim",
    label_col: str = "ground_truth",
) -> plt.Figure:
    """Plot genuine vs impostor score distribution as histograms.

    df: results DataFrame with a score column and binary ground_truth (1=genuine, 0=impostor)
    """
    if score_col not in df.columns or label_col not in df.columns:
        return plt.Figure()

    scores = df[score_col].dropna()
    labels = df[label_col].dropna().astype(int).values
    valid = np.isfinite(scores.values) & (labels >= 0)

    if valid.sum() < 2:
        return plt.Figure()

    scores = scores.values[valid]
    labels = labels[valid]

    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(genuine, color="blue", alpha=0.5, bins=50, density=True, label="Genuine")
    ax.hist(impostor, color="red", alpha=0.5, bins=50, density=True, label="Impostor")
    ax.set_xlim(0, 1)
    ax.set_title(f"Score Distribution — {eval_name} | {dataset} / {technique}")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_score_summary(
    results: dict,  # {technique: {dataset: {eval: csv_path}}}
) -> plt.Figure:
    """Create a table of mean ± std scores across all evaluations.

    results: nested dict of technique -> dataset -> eval -> csv_path
    Returns (figure, stats_df) — stats_df is a DataFrame suitable for display.
    """
    # Collect stats
    stats_rows = []
    for tech, ds_dict in sorted(results.items()):
        for ds_name, eval_dict in sorted(ds_dict.items()):
            for ev_name, csv_path in sorted(eval_dict.items()):
                try:
                    df = pd.read_csv(csv_path)
                    scores = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                    mean_val = scores.mean()
                    std_val = scores.std() if len(scores) > 1 else 0.0
                    stats_rows.append({
                        "technique": tech,
                        "dataset": ds_name,
                        "metric": ev_name,
                        "mean": f"{mean_val:.2f}",
                        "std": f"{std_val:.2f}",
                        "mean_std": f"{mean_val:.2f} ± {std_val:.2f}" if std_val > 0 else f"{mean_val:.2f}",
                        "count": len(scores),
                    })
                except Exception:
                    stats_rows.append({
                        "technique": tech,
                        "dataset": ds_name,
                        "metric": ev_name,
                        "mean": "/",
                        "std": "/",
                        "mean_std": "/",
                        "count": 0,
                    })

    stats_df = pd.DataFrame(stats_rows)

    # Create pivot for table display
    if stats_df.empty:
        return plt.Figure(), stats_df

    fig, ax = plt.subplots(figsize=(14, max(4, len(stats_df) * 0.4)))
    ax.axis("tight")
    ax.axis("off")

    # Build table data: rows=technique/dataset, cols=metrics
    table_data = []
    for tech in sorted(results.keys()):
        for ds_name in sorted(results[tech].keys()):
            row = {"Technique": tech, "Dataset": ds_name}
            for ev_name in sorted(results[tech][ds_name].keys()):
                csv_path = results[tech][ds_name][ev_name]
                try:
                    df = pd.read_csv(csv_path)
                    scores = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                    row[ev_name] = f"{scores.mean():.2f}±{scores.std():.2f}" if len(scores) > 1 else f"{scores.mean():.2f}"
                except Exception:
                    row[ev_name] = "/"
            table_data.append(row)

    display_df = pd.DataFrame(table_data)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"] * len(display_df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.set_title("Score Summary (mean ± std)", pad=20)
    fig.tight_layout()

    return fig, stats_df


def _get_eval_category(eval_name: str) -> str | None:
    """Map an evaluation name to a radar axis category."""
    name = eval_name.lower()
    if any(k in name for k in ("adaface", "swinface", "vggface", "deid", "div", "privacy", "anonym", "identit")):
        return "Privacy Protection"
    if any(k in name for k in ("emotion", "gaze", "gender", "age", "expression", "classif", "util", "dan", "gd")):
        return "Data Utility"
    if any(k in name for k in ("ssim", "lpips", "mse", "fid", "quality", "luminance")):
        return "Image Quality"
    if any(k in name for k in ("context", "pose", "attribute", "background", "scene")):
        return "Context Preservation"
    if any(k in name for k in ("auc", "eer", "verification", "roc", "pr", "precision", "recall", "f1", "accuracy")):
        return "Verification"
    return None


def plot_radar_chart(
    results: dict,  # {technique: {dataset: {eval_name: csv_path}}}
    dataset: str,
    selected_techniques: list[str] | None = None,
) -> plt.Figure:
    """Create a radar chart comparing techniques across aggregated metric categories.

    Groups available evaluation metrics into axis categories (Privacy, Image Quality,
    Data Utility, etc.), normalizes scores to 0-1, and plots one polygon per technique.
    """
    import numpy as np

    if selected_techniques is None:
        selected_techniques = sorted(results.keys())

    # Step 1: Load scores per technique
    # metrics_by_tech[technique][eval_name] = raw_or_clamped_score
    metrics_by_tech: dict[str, dict[str, float]] = {t: {} for t in selected_techniques}

    for tech in selected_techniques:
        if tech not in results:
            continue
        ds_scores = results[tech].get(dataset, {})
        for eval_name, csv_path in ds_scores.items():
            try:
                df = pd.read_csv(csv_path)
                score_col = None
                for candidate in ["score", "cosine_similarity", "cossim", "similarity_score", "similarity"]:
                    if candidate in df.columns:
                        score_col = candidate
                        break
                if not score_col:
                    score_candidates = [c for c in df.columns if "score" in c.lower()]
                    score_col = score_candidates[0] if score_candidates else None
                if score_col is None:
                    score_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                mean_raw = float(df[score_col].dropna().mean())
            except Exception:
                mean_raw = np.nan
                continue

            # Clamp bounded metrics (SSIM, cossim) to [0, 1].
            # Leave unbounded metrics (FID, MSE) raw — per-metric normalization handles it.
            if score_col and any(k in score_col for k in ("ssim", "cossim", "accuracy", "precision", "recall", "f1")):
                metrics_by_tech[tech][eval_name] = min(max(mean_raw, 0.0), 1.0)
            else:
                metrics_by_tech[tech][eval_name] = mean_raw

    # Step 2: Normalize each metric across all techniques to [0, 1] first.
    #         This ensures metrics with different raw ranges contribute equally.
    all_metrics = list(metrics_by_tech[selected_techniques[0]].keys())
    if not all_metrics:
        return plt.Figure()

    metric_norm: dict[str, tuple[float, float]] = {}  # eval_name -> (min, max)
    for met in all_metrics:
        vals = [metrics_by_tech[t][met] for t in selected_techniques if met in metrics_by_tech[t]]
        if len(vals) < 2:
            metric_norm[met] = (0.0, 1.0)
            continue
        min_v, max_v = min(vals), max(vals)
        metric_norm[met] = (min_v, max_v if max_v != min_v else min_v + 1.0)

    # Normalize and group into categories
    # Higher raw score should mean better performance for the radar chart.
    # For inverted metrics (MSE, FID, LPIPS), we flip: 1.0 - norm.
    def _is_inverted(eval_name: str) -> bool:
        name = eval_name.lower()
        return any(k in name for k in ("mse", "fid", "lpips", "loss", "distance", "error"))

    cats_by_tech: dict[str, dict[str, float]] = {t: {} for t in selected_techniques}
    for tech in selected_techniques:
        for eval_name, raw_score in metrics_by_tech[tech].items():
            cat = _get_eval_category(eval_name)
            if cat is None:
                continue
            min_v, max_v = metric_norm.get(eval_name, (0.0, 1.0))
            norm_score = (raw_score - min_v) / (max_v - min_v)
            if _is_inverted(eval_name):
                norm_score = 1.0 - norm_score
            if cat not in cats_by_tech[tech]:
                cats_by_tech[tech][cat] = []
            cats_by_tech[tech][cat].append(norm_score)

    for tech in selected_techniques:
        for cat in cats_by_tech[tech]:
            vals = cats_by_tech[tech][cat]
            cats_by_tech[tech][cat] = sum(vals) / len(vals) if vals else 0.0

    all_categories = sorted({cat for tech in selected_techniques for cat in cats_by_tech[tech]})
    if not all_categories:
        return plt.Figure()

    # Step 4: Plot
    N = len(all_categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_categories, fontsize=10)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

    colors = sns.color_palette("husl", len(selected_techniques))
    for (tech, color) in zip(selected_techniques, colors):
        values = [cats_by_tech[tech].get(cat, 0.0) for cat in all_categories] + [cats_by_tech[tech].get(all_categories[0], 0.0)]
        ax.plot(angles, values, linewidth=2, color=color, label=tech)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.suptitle(f"Radar Comparison — {dataset}", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_roc_multi(
    dfs: dict[str, pd.DataFrame],  # technique -> DataFrame with score + ground_truth columns
    dataset: str,
    eval_name: str,
    max_fpr_steps: int = 100,
) -> tuple[plt.Figure, list[dict]]:
    """Plot multiple ROC curves on one axis for comparing techniques.

    Returns (figure, list of dicts with AUC/EER/best_threshold info).
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc  # type: ignore

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("husl", len(dfs))
    info = []

    for (tech, df), color in zip(dfs.items(), colors):
        score_col, label_col = _find_score_and_label_cols(df)
        if score_col is None or label_col is None:
            continue

        labels = df[label_col].dropna().astype(int).values
        scores = df[score_col].dropna().values
        valid = labels >= 0
        if valid.sum() < 2:
            continue

        labels = labels[valid]
        scores = scores[valid]

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # EER
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        eer_idx = np.argmin(abs_diffs)
        eer = float(np.mean((fpr[eer_idx], fnr[eer_idx])))
        best_thresh = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.0

        # Smooth with interpolation
        mean_fpr = np.linspace(0, 1, max_fpr_steps)
        mean_tpr = np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        std_tpr = np.std([tpr] * 1, axis=0)  # single fold — use 0 as baseline

        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{tech} (AUC={roc_auc:.3f}, EER={eer:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.1)

        # Mark EER point
        ax.scatter([eer], [eer], color=color, s=60, zorder=5)

        info.append({"technique": tech, "auc": roc_auc, "eer": eer, "threshold": best_thresh})

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Comparison — {eval_name} | {dataset}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, info
