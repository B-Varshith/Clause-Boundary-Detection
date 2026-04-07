"""
crf_model.py - CRF (Conditional Random Field) model for clause boundary detection.

Uses sklearn-crfsuite for sequence labeling with BIO tags to detect clause boundaries.
Features are extracted using feature_extractor.py.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

from src.feature_extractor import sent2features, sent2labels, sent2tokens


class CRFClauseDetector:
    """CRF-based clause boundary detector."""

    def __init__(self, c1: float = 0.1, c2: float = 0.1,
                 max_iterations: int = 100, algorithm: str = "lbfgs"):
        """
        Initialize the CRF model.

        Args:
            c1: L1 regularization coefficient
            c2: L2 regularization coefficient
            max_iterations: Maximum number of training iterations
            algorithm: Training algorithm ('lbfgs', 'l2sgd', etc.)
        """
        self.crf = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=True,
        )
        self.is_trained = False

    def train(self, train_data: List[List[Tuple]],
              dev_data: Optional[List[List[Tuple]]] = None) -> Dict[str, Any]:
        """
        Train the CRF model.

        Args:
            train_data: Training data - list of sentences, each sentence is
                       list of (word, upos, deprel, xpos, head_dist, label) tuples
            dev_data: Optional dev data for evaluation during training

        Returns:
            Dictionary with training results
        """
        print("Extracting features for training data...")
        X_train = [sent2features(s) for s in train_data]
        y_train = [sent2labels(s) for s in train_data]

        print(f"Training CRF on {len(X_train)} sentences...")
        self.crf.fit(X_train, y_train)
        self.is_trained = True

        results = {
            "num_train_sentences": len(X_train),
            "labels": list(self.crf.classes_),
        }

        # Evaluate on dev set if provided
        if dev_data:
            print("Evaluating on dev set...")
            dev_results = self.evaluate(dev_data)
            results["dev_metrics"] = dev_results

        return results

    def predict(self, data: List[List[Tuple]]) -> List[List[str]]:
        """
        Predict BIO labels for sentences.

        Args:
            data: List of sentences in the same tuple format as training

        Returns:
            List of predicted label sequences
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X = [sent2features(s) for s in data]
        return self.crf.predict(X)

    def predict_single(self, sentence: List[Tuple]) -> List[str]:
        """
        Predict BIO labels for a single sentence.

        Args:
            sentence: List of (word, upos, deprel, xpos, head_dist) tuples
                     (no label needed)

        Returns:
            List of predicted BIO labels
        """
        # Add dummy labels for feature extraction compatibility
        sent_with_dummy = [(w, u, d, x, h, "O") for w, u, d, x, h in sentence]
        X = [sent2features(sent_with_dummy)]
        return self.crf.predict(X)[0]

    def evaluate(self, test_data: List[List[Tuple]]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test data in the same format as training data

        Returns:
            Dictionary with precision, recall, F1 for each label
        """
        X_test = [sent2features(s) for s in test_data]
        y_test = [sent2labels(s) for s in test_data]
        y_pred = self.crf.predict(X_test)

        # Labels to evaluate (exclude 'O')
        labels = [l for l in self.crf.classes_ if l != "O"]

        report = crf_metrics.flat_classification_report(
            y_test, y_pred, labels=labels, output_dict=True
        )

        # Also compute overall metrics
        from sklearn.metrics import classification_report
        y_test_flat = [label for sent in y_test for label in sent]
        y_pred_flat = [label for sent in y_pred for label in sent]

        overall = classification_report(
            y_test_flat, y_pred_flat,
            labels=["B-CLAUSE", "I-CLAUSE", "O"],
            output_dict=True
        )

        return {
            "per_label": report,
            "overall": overall,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    def get_top_features(self, n: int = 20) -> Dict[str, List]:
        """
        Get the top positive and negative features for each label.

        Args:
            n: Number of top features to return

        Returns:
            Dictionary mapping label -> list of (feature, weight) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        top_features = {}
        for label in self.crf.classes_:
            if label == "O":
                continue

            state_features = self.crf.state_features_
            label_features = {
                attr: weight
                for (attr, lbl), weight in state_features.items()
                if lbl == label
            }

            # Sort by absolute weight
            sorted_features = sorted(
                label_features.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:n]

            top_features[label] = sorted_features

        return top_features

    def save(self, filepath: str):
        """Save the trained model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.crf, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, "rb") as f:
            self.crf = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Quick test with dummy data
    sample_train = [
        [
            ("When", "SCONJ", "mark", "WRB", 2, "B-CLAUSE"),
            ("the", "DET", "det", "DT", 1, "I-CLAUSE"),
            ("rain", "NOUN", "nsubj", "NN", 1, "I-CLAUSE"),
            ("stopped", "VERB", "advcl", "VBD", 3, "I-CLAUSE"),
            (",", "PUNCT", "punct", ",", 3, "O"),
            ("we", "PRON", "nsubj", "PRP", 1, "B-CLAUSE"),
            ("went", "VERB", "root", "VBD", 0, "I-CLAUSE"),
            ("outside", "ADV", "advmod", "RB", 1, "I-CLAUSE"),
        ]
    ] * 10  # Repeat for minimum training

    model = CRFClauseDetector(max_iterations=50)
    results = model.train(sample_train)
    print(f"Training complete. Labels: {results['labels']}")

    # Predict
    predictions = model.predict(sample_train[:1])
    print(f"Predictions: {predictions[0]}")
