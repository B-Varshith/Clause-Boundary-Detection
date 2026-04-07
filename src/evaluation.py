"""
evaluation.py - Evaluation metrics and analysis for clause boundary detection.

Computes Precision, Recall, F1 at both token-level and clause-level,
generates confusion matrices, and performs error analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import Counter

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)


def compute_token_metrics(y_true: List[List[str]],
                           y_pred: List[List[str]],
                           labels: List[str] = None) -> Dict[str, Any]:
    """
    Compute token-level classification metrics.

    Args:
        y_true: Ground truth labels (list of sentence label lists)
        y_pred: Predicted labels (list of sentence label lists)
        labels: Labels to evaluate (default: B-CLAUSE, I-CLAUSE, O)

    Returns:
        Dictionary with per-label and overall metrics
    """
    if labels is None:
        labels = ["B-CLAUSE", "I-CLAUSE", "O"]

    # Flatten
    y_true_flat = [l for sent in y_true for l in sent]
    y_pred_flat = [l for sent in y_pred for l in sent]

    # Classification report
    report = classification_report(
        y_true_flat, y_pred_flat,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    # Clause-specific metrics (excluding O)
    clause_labels = [l for l in labels if l != "O"]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat,
        labels=[labels.index(l) if isinstance(l, str) else l for l in clause_labels],
        average="macro",
        zero_division=0,
    )

    # Recompute with string labels
    clause_f1 = f1_score(
        y_true_flat, y_pred_flat,
        labels=clause_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "per_label": report,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "clause_macro_f1": clause_f1,
        "total_tokens": len(y_true_flat),
    }


def compute_clause_metrics(y_true: List[List[str]],
                            y_pred: List[List[str]]) -> Dict[str, float]:
    """
    Compute clause-level exact match metrics.
    A clause is considered correctly detected if all its tokens have the
    correct BIO labels.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with clause-level precision, recall, F1
    """
    true_clauses = _extract_clause_spans_from_bio(y_true)
    pred_clauses = _extract_clause_spans_from_bio(y_pred)

    # Count matches
    tp = 0
    for sent_true, sent_pred in zip(true_clauses, pred_clauses):
        for clause in sent_true:
            if clause in sent_pred:
                tp += 1

    total_true = sum(len(s) for s in true_clauses)
    total_pred = sum(len(s) for s in pred_clauses)

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_true if total_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "clause_precision": round(precision, 4),
        "clause_recall": round(recall, 4),
        "clause_f1": round(f1, 4),
        "true_clauses": total_true,
        "pred_clauses": total_pred,
        "correct_clauses": tp,
    }


def _extract_clause_spans_from_bio(bio_sequences: List[List[str]]) -> List[List[Tuple[int, int]]]:
    """
    Extract clause spans from BIO label sequences.

    Returns:
        List of sentence-level clause span lists, each span is (start, end)
    """
    all_spans = []
    for labels in bio_sequences:
        spans = []
        start = None
        for i, label in enumerate(labels):
            if label == "B-CLAUSE":
                if start is not None:
                    spans.append((start, i - 1))
                start = i
            elif label == "O":
                if start is not None:
                    spans.append((start, i - 1))
                    start = None
        if start is not None:
            spans.append((start, len(labels) - 1))
        all_spans.append(spans)
    return all_spans


def error_analysis(y_true: List[List[str]],
                    y_pred: List[List[str]],
                    sentences: List[List[str]]) -> Dict[str, Any]:
    """
    Perform error analysis on predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sentences: Token lists for each sentence

    Returns:
        Dictionary with error statistics and examples
    """
    errors = {
        "total_errors": 0,
        "error_types": Counter(),
        "errors_by_sentence_length": {"short": 0, "medium": 0, "long": 0},
        "sample_errors": [],
    }

    for sent_idx, (true_seq, pred_seq) in enumerate(zip(y_true, y_pred)):
        sent_tokens = sentences[sent_idx] if sent_idx < len(sentences) else []
        sent_len = len(true_seq)

        for i, (t, p) in enumerate(zip(true_seq, pred_seq)):
            if t != p:
                errors["total_errors"] += 1
                error_type = f"{t} -> {p}"
                errors["error_types"][error_type] += 1

                # Categorize by sentence length
                if sent_len <= 10:
                    errors["errors_by_sentence_length"]["short"] += 1
                elif sent_len <= 25:
                    errors["errors_by_sentence_length"]["medium"] += 1
                else:
                    errors["errors_by_sentence_length"]["long"] += 1

                # Collect sample errors (limited)
                if len(errors["sample_errors"]) < 20:
                    token = sent_tokens[i] if i < len(sent_tokens) else "?"
                    context_start = max(0, i - 2)
                    context_end = min(len(sent_tokens), i + 3)
                    context = sent_tokens[context_start:context_end] if sent_tokens else []
                    errors["sample_errors"].append({
                        "sentence_idx": sent_idx,
                        "token_idx": i,
                        "token": token,
                        "true_label": t,
                        "pred_label": p,
                        "context": context,
                    })

    errors["error_types"] = dict(errors["error_types"].most_common())
    return errors


def compare_models(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare metrics across multiple models.

    Args:
        results: Dict mapping model_name -> evaluation results dict

    Returns:
        Comparison summary
    """
    comparison = {"models": []}

    for model_name, metrics in results.items():
        if "per_label" in metrics:
            report = metrics["per_label"]
        elif "overall" in metrics and "per_label" in metrics["overall"]:
            report = metrics["overall"]["per_label"]
        elif "overall" in metrics:
            report = metrics["overall"]
        else:
            report = metrics

        b_clause = report.get("B-CLAUSE", {})
        i_clause = report.get("I-CLAUSE", {})

        comparison["models"].append({
            "name": model_name,
            "B-CLAUSE_precision": b_clause.get("precision", 0),
            "B-CLAUSE_recall": b_clause.get("recall", 0),
            "B-CLAUSE_f1": b_clause.get("f1-score", 0),
            "I-CLAUSE_precision": i_clause.get("precision", 0),
            "I-CLAUSE_recall": i_clause.get("recall", 0),
            "I-CLAUSE_f1": i_clause.get("f1-score", 0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
            "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0),
        })

    return comparison


def save_results(results: Dict[str, Any], filepath: str):
    """Save evaluation results to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(results, default=convert))

    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Results saved to {filepath}")


def print_results_table(results: Dict[str, Any], model_name: str = "Model"):
    """Print formatted results table."""
    report = results.get("per_label", results.get("overall", {}))

    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*60}")

    for label in ["B-CLAUSE", "I-CLAUSE", "O"]:
        if label in report:
            m = report[label]
            print(f"{label:<15} {m.get('precision',0):<12.4f} {m.get('recall',0):<12.4f} "
                  f"{m.get('f1-score',0):<12.4f} {m.get('support',0):<10}")

    if "macro avg" in report:
        m = report["macro avg"]
        print(f"{'-'*60}")
        print(f"{'Macro Avg':<15} {m.get('precision',0):<12.4f} {m.get('recall',0):<12.4f} "
              f"{m.get('f1-score',0):<12.4f} {m.get('support',0):<10}")

    if "weighted avg" in report:
        m = report["weighted avg"]
        print(f"{'Weighted Avg':<15} {m.get('precision',0):<12.4f} {m.get('recall',0):<12.4f} "
              f"{m.get('f1-score',0):<12.4f} {m.get('support',0):<10}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Quick test
    y_true = [
        ["B-CLAUSE", "I-CLAUSE", "I-CLAUSE", "I-CLAUSE", "O", "B-CLAUSE", "I-CLAUSE", "I-CLAUSE"],
        ["B-CLAUSE", "I-CLAUSE", "I-CLAUSE", "O", "B-CLAUSE", "I-CLAUSE"],
    ]
    y_pred = [
        ["B-CLAUSE", "I-CLAUSE", "I-CLAUSE", "O", "O", "B-CLAUSE", "I-CLAUSE", "I-CLAUSE"],
        ["B-CLAUSE", "I-CLAUSE", "I-CLAUSE", "O", "B-CLAUSE", "I-CLAUSE"],
    ]

    token_metrics = compute_token_metrics(y_true, y_pred)
    print_results_table(token_metrics, "Test Model")

    clause_metrics = compute_clause_metrics(y_true, y_pred)
    print(f"Clause-level: {clause_metrics}")
