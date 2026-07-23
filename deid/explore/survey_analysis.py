"""Survey analysis — computes human verification metrics."""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

SURVEY_DIR = Path(__file__).parent.parent.parent / "results" / "human_survey"


def compute_metrics(responses: list[dict]) -> dict:
    """Compute human verification metrics from survey responses.

    Args:
        responses: List of survey response dictionaries

    Returns:
        Metrics dictionary
    """
    # Aggregate data
    technique_metrics = defaultdict(lambda: {
        "same": {"total": 0, "correct": 0, "confidences": []},
        "different": {"total": 0, "correct": 0, "confidences": []},
    })

    attribute_metrics = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "correct": 0, "confidences": []
    }))

    dataset_metrics = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "correct": 0
    }))

    total_responses = 0
    total_correct = 0

    for response in responses:
        for pair in response.get("responses", []):
            technique = pair.get("technique", "unknown")
            dataset = pair.get("dataset", "unknown")
            pair_type = pair.get("pair_type", "")
            answer = pair.get("answer", "")
            confidence = pair.get("confidence", 3)

            # Check if answer is correct
            is_correct = (pair_type == answer)
            total_responses += 1
            if is_correct:
                total_correct += 1

            # Aggregate by technique
            tm = technique_metrics[technique]
            tm[pair_type]["total"] += 1
            if is_correct:
                tm[pair_type]["correct"] += 1
            tm[pair_type]["confidences"].append(confidence)

            # Aggregate by technique + attribute
            for attr in pair.get("labels", {}):
                attr_val = pair["labels"][attr]
                am = attribute_metrics[technique][attr][attr_val]
                am["total"] += 1
                if is_correct:
                    am["correct"] += 1
                am["confidences"].append(confidence)

            # Aggregate by dataset
            dataset_metrics[dataset][pair_type]["total"] += 1
            if is_correct:
                dataset_metrics[dataset][pair_type]["correct"] += 1

    # Calculate final metrics
    metrics = {
        "total_responses": total_responses,
        "total_correct": total_correct,
        "overall_accuracy": total_correct / total_responses if total_responses > 0 else 0,
        "by_technique": {},
        "by_dataset": {},
    }

    for technique, tm in technique_metrics.items():
        same_acc = tm["same"]["correct"] / tm["same"]["total"] if tm["same"]["total"] > 0 else 0
        diff_acc = tm["different"]["correct"] / tm["different"]["total"] if tm["different"]["total"] > 0 else 0
        avg_same_conf = (sum(tm["same"]["confidences"]) / len(tm["same"]["confidences"]) if tm["same"]["confidences"] else 0)
        avg_diff_conf = (sum(tm["different"]["confidences"]) / len(tm["different"]["confidences"]) if tm["different"]["confidences"] else 0)

        metrics["by_technique"][technique] = {
            "same_pairs": {
                "total": tm["same"]["total"],
                "accuracy": same_acc,
                "avg_confidence": avg_same_conf,
            },
            "different_pairs": {
                "total": tm["different"]["total"],
                "accuracy": diff_acc,
                "avg_confidence": avg_diff_conf,
            },
            "overall_accuracy": (same_acc + diff_acc) / 2,
        }

    for dataset, dm in dataset_metrics.items():
        same_acc = dm["same"]["correct"] / dm["same"]["total"] if dm["same"]["total"] > 0 else 0
        diff_acc = dm["different"]["correct"] / dm["different"]["total"] if dm["different"]["total"] > 0 else 0
        metrics["by_dataset"][dataset] = {
            "same_pairs": {
                "total": dm["same"]["total"],
                "accuracy": same_acc,
            },
            "different_pairs": {
                "total": dm["different"]["total"],
                "accuracy": diff_acc,
            },
            "overall_accuracy": (same_acc + diff_acc) / 2,
        }

    return metrics


def export_to_csv(metrics: dict, output_file: Path) -> None:
    """Export metrics to CSV file."""
    import csv

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write overall metrics
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Responses", metrics["total_responses"]])
        writer.writerow(["Overall Accuracy", f"{metrics['overall_accuracy']:.2%}"])
        writer.writerow([])

        # Write per-technique metrics
        writer.writerow(["Technique", "Pair Type", "Total", "Accuracy", "Avg Confidence"])
        for technique, tm in metrics["by_technique"].items():
            writer.writerow([technique, "same", tm["same_pairs"]["total"],
                           f"{tm['same_pairs']['accuracy']:.2%}",
                           f"{tm['same_pairs']['avg_confidence']:.2f}"])
            writer.writerow([technique, "different", tm["different_pairs"]["total"],
                           f"{tm['different_pairs']['accuracy']:.2%}",
                           f"{tm['different_pairs']['avg_confidence']:.2f}"])


def load_and_analyze() -> dict:
    """Load responses and compute metrics."""
    if not SURVEY_DIR.exists():
        return {"error": "No survey data found"}

    responses = []
    for file in SURVEY_DIR.glob("response_*.json"):
        with open(file, "r") as f:
            responses.append(json.load(f))

    if not responses:
        return {"error": "No responses found"}

    return compute_metrics(responses)
