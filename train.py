import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset, extract_tokens, get_dataset_stats
from src.clause_labeler import generate_labels_for_dataset, generate_bio_labels
from src.feature_extractor import prepare_crf_data, sent2features, sent2labels
from src.evaluation import (
    compute_token_metrics,
    compute_clause_metrics,
    error_analysis,
    save_results,
    print_results_table,
    compare_models,
)


def prepare_data(data_dir: str = "data/UD_English-EWT"):
    """Load and preprocess the dataset."""
    print("=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)

    dataset = load_dataset(data_dir)
    stats = get_dataset_stats(dataset)

    print("\nDataset Statistics:")
    for split, s in stats.items():
        print(f"  {split}: {s}")

    # Generate BIO labels
    print("\nGenerating BIO clause labels...")
    labeled = {}
    tokens = {}
    for split in ["train", "dev", "test"]:
        labeled[split] = generate_labels_for_dataset(dataset[split])
        tokens[split] = [extract_tokens(s) for s in dataset[split]]
        print(f"  {split}: {len(labeled[split])} labeled sentences")

    # Prepare CRF-format data
    print("\nPreparing CRF-format data...")
    crf_data = {}
    for split in ["train", "dev", "test"]:
        # Align labeled sentences with token data
        min_len = min(len(labeled[split]), len(tokens[split]))
        crf_data[split] = prepare_crf_data(
            labeled[split][:min_len],
            tokens[split][:min_len]
        )
        print(f"  {split}: {len(crf_data[split])} sentences prepared")

    return dataset, labeled, tokens, crf_data


def train_rule_based(labeled, tokens):
    """Evaluate the rule-based system."""
    print("\n" + "=" * 60)
    print("  RULE-BASED MODEL")
    print("=" * 60)

    from src.rule_based import predict_bio_for_tokens

    # Predict on test set
    print("Predicting on test set...")
    y_true = []
    y_pred = []
    test_tokens = []

    for sent_labels in labeled["test"]:
        words = [w for w, _, _ in sent_labels]
        true_labels = [l for _, _, l in sent_labels]

        pred_labels = predict_bio_for_tokens(words)

        # Ensure same length
        min_len = min(len(true_labels), len(pred_labels))
        y_true.append(true_labels[:min_len])
        y_pred.append(pred_labels[:min_len])
        test_tokens.append(words[:min_len])

    # Evaluate
    token_metrics = compute_token_metrics(y_true, y_pred)
    clause_metrics = compute_clause_metrics(y_true, y_pred)
    errors = error_analysis(y_true, y_pred, test_tokens)

    print_results_table(token_metrics, "Rule-Based")
    print(f"Clause-level F1: {clause_metrics['clause_f1']:.4f}")
    print(f"Total errors: {errors['total_errors']}")

    # Save results
    results = {
        **token_metrics,
        "clause_metrics": clause_metrics,
        "error_analysis": {
            "total_errors": errors["total_errors"],
            "error_types": errors["error_types"],
            "errors_by_length": errors["errors_by_sentence_length"],
        }
    }
    save_results(results, "results/metrics/rule_based_results.json")

    return results


def train_crf(crf_data):
    """Train and evaluate the CRF model."""
    print("\n" + "=" * 60)
    print("  CRF MODEL")
    print("=" * 60)

    from src.crf_model import CRFClauseDetector

    model = CRFClauseDetector(c1=0.1, c2=0.1, max_iterations=100)

    # Train
    start_time = time.time()
    train_results = model.train(crf_data["train"], dev_data=crf_data["dev"])
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f}s")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(crf_data["test"])

    print_results_table(test_results["overall"], "CRF")

    # Clause-level metrics
    clause_metrics = compute_clause_metrics(
        test_results["y_test"], test_results["y_pred"]
    )
    print(f"Clause-level F1: {clause_metrics['clause_f1']:.4f}")

    # Feature importance
    print("\nTop features for B-CLAUSE:")
    top_features = model.get_top_features(n=10)
    for label, features in top_features.items():
        print(f"\n  {label}:")
        for feat, weight in features[:10]:
            print(f"    {feat}: {weight:.4f}")

    # Save model and results
    model.save("results/models/crf_model.pkl")

    results = {
        **test_results["overall"],
        "clause_metrics": clause_metrics,
        "training_time": train_time,
    }
    # Remove non-serializable items
    save_results(results, "results/metrics/crf_results.json")

    return results


def train_bilstm(crf_data):
    """Train and evaluate the BiLSTM model."""
    print("\n" + "=" * 60)
    print("  BiLSTM MODEL")
    print("=" * 60)

    from src.bilstm_model import BiLSTMTrainer

    trainer = BiLSTMTrainer(
        word_emb_dim=100,
        pos_emb_dim=25,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        lr=0.001,
        batch_size=32,
    )

    # Train
    start_time = time.time()
    history = trainer.train(
        crf_data["train"],
        dev_data=crf_data["dev"],
        epochs=20,
        patience=5,
    )
    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Best dev F1: {history.get('best_dev_f1', 0):.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(crf_data["test"])

    print_results_table(test_results["overall"], "BiLSTM")

    # Clause-level metrics
    clause_metrics = compute_clause_metrics(
        test_results["y_test"], test_results["y_pred"]
    )
    print(f"Clause-level F1: {clause_metrics['clause_f1']:.4f}")

    # Save model and results
    trainer.save("results/models/bilstm")

    results = {
        **test_results["overall"],
        "clause_metrics": clause_metrics,
        "training_time": train_time,
        "training_history": {
            "train_loss": history["train_loss"],
            "dev_f1": history.get("dev_f1", []),
            "best_dev_f1": history.get("best_dev_f1", 0),
        },
    }
    save_results(results, "results/metrics/bilstm_results.json")

    return results


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Clause Boundary Detection - Training Pipeline       ║")
    print("║     Universal Dependencies English-EWT Dataset          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Load and prepare data
    dataset, labeled, tokens, crf_data = prepare_data()

    # Step 2: Train and evaluate each model
    all_results = {}

    # Rule-based
    try:
        all_results["rule_based"] = train_rule_based(labeled, tokens)
    except Exception as e:
        print(f"Rule-based model failed: {e}")
        import traceback
        traceback.print_exc()

    # CRF
    try:
        all_results["crf"] = train_crf(crf_data)
    except Exception as e:
        print(f"CRF model failed: {e}")
        import traceback
        traceback.print_exc()

    # BiLSTM
    try:
        all_results["bilstm"] = train_bilstm(crf_data)
    except Exception as e:
        print(f"BiLSTM model failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 3: Compare all models
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    if all_results:
        comparison = compare_models(all_results)
        print(f"\n{'Model':<20} {'Macro F1':<12} {'B-CLAUSE F1':<15} {'I-CLAUSE F1':<15}")
        print("-" * 62)
        for model in comparison["models"]:
            print(f"{model['name']:<20} {model['macro_f1']:<12.4f} "
                  f"{model['B-CLAUSE_f1']:<15.4f} {model['I-CLAUSE_f1']:<15.4f}")

        save_results(comparison, "results/metrics/comparison.json")

    print("\n✅ Training pipeline complete!")
    print("   Results saved to results/metrics/")
    print("   Models saved to results/models/")
    print("   Run 'streamlit run app.py' to launch the visualization app")


if __name__ == "__main__":
    main()
