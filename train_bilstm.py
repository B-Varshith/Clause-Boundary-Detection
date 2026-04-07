import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset, extract_tokens, get_dataset_stats
from src.clause_labeler import generate_labels_for_dataset
from src.feature_extractor import prepare_crf_data
from src.evaluation import (
    compute_token_metrics,
    compute_clause_metrics,
    save_results,
    print_results_table,
    compare_models,
)


def main():
    print("=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)

    dataset = load_dataset("data/UD_English-EWT")

    print("\nGenerating BIO clause labels...")
    labeled = {}
    tokens = {}
    for split in ["train", "dev", "test"]:
        labeled[split] = generate_labels_for_dataset(dataset[split])
        tokens[split] = [extract_tokens(s) for s in dataset[split]]
        print(f"  {split}: {len(labeled[split])} labeled sentences")

    print("\nPreparing CRF-format data...")
    crf_data = {}
    for split in ["train", "dev", "test"]:
        min_len = min(len(labeled[split]), len(tokens[split]))
        crf_data[split] = prepare_crf_data(
            labeled[split][:min_len],
            tokens[split][:min_len]
        )
        print(f"  {split}: {len(crf_data[split])} sentences prepared")

    # ============================
    # Train BiLSTM
    # ============================
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

    # Save model
    trainer.save("results/models/bilstm")

    # Save results
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

    # ============================
    # Update comparison with all 3 models
    # ============================
    print("\n" + "=" * 60)
    print("  UPDATING MODEL COMPARISON")
    print("=" * 60)

    # Load existing results
    all_results = {}
    
    rb_path = Path("results/metrics/rule_based_results.json")
    if rb_path.exists():
        with open(rb_path) as f:
            all_results["rule_based"] = json.load(f)

    crf_path = Path("results/metrics/crf_results.json")
    if crf_path.exists():
        with open(crf_path) as f:
            all_results["crf"] = json.load(f)

    all_results["bilstm"] = results

    comparison = compare_models(all_results)
    print(f"\n{'Model':<20} {'Macro F1':<12} {'B-CLAUSE F1':<15} {'I-CLAUSE F1':<15}")
    print("-" * 62)
    for model in comparison["models"]:
        print(f"{model['name']:<20} {model['macro_f1']:<12.4f} "
              f"{model['B-CLAUSE_f1']:<15.4f} {model['I-CLAUSE_f1']:<15.4f}")

    save_results(comparison, "results/metrics/comparison.json")

    print("\n✅ BiLSTM training and full comparison complete!")
    print("   Run 'streamlit run app.py' to launch the visualization app")


if __name__ == "__main__":
    main()
