"""
bilstm_model.py - BiLSTM (Bidirectional LSTM) model for clause boundary detection.

PyTorch implementation of a BiLSTM sequence labeling model with word + POS embeddings
for predicting BIO clause boundary tags.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm


# ==================== Vocabulary ====================

class Vocabulary:
    """Maps tokens/labels to integer indices."""

    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token2idx = {pad_token: 0, unk_token: 1}
        self.idx2token = {0: pad_token, 1: unk_token}
        self.counter = Counter()

    def add_token(self, token: str):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        self.counter[token] += 1

    def build_from_data(self, tokens: List[str], min_freq: int = 1):
        counter = Counter(tokens)
        for token, freq in counter.items():
            if freq >= min_freq:
                self.add_token(token)

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def decode(self, idx: int) -> str:
        return self.idx2token.get(idx, self.unk_token)

    def __len__(self):
        return len(self.token2idx)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump({"token2idx": self.token2idx}, f)

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.token2idx = data["token2idx"]
        self.idx2token = {int(v): k for k, v in self.token2idx.items()}


# ==================== Dataset ====================

class ClauseDataset(Dataset):
    """PyTorch dataset for clause boundary detection."""

    def __init__(self, sentences: List[List[Tuple]],
                 word_vocab: Vocabulary,
                 pos_vocab: Vocabulary,
                 label_vocab: Vocabulary):
        """
        Args:
            sentences: List of sentences, each is list of
                      (word, upos, deprel, xpos, head_dist, label) tuples
            word_vocab: Vocabulary for words
            pos_vocab: Vocabulary for POS tags
            label_vocab: Vocabulary for BIO labels
        """
        self.word_ids = []
        self.pos_ids = []
        self.label_ids = []
        self.lengths = []

        for sent in sentences:
            words = [word_vocab.encode(t[0].lower()) for t in sent]
            pos = [pos_vocab.encode(t[1]) for t in sent]
            labels = [label_vocab.encode(t[-1]) for t in sent]

            self.word_ids.append(torch.tensor(words, dtype=torch.long))
            self.pos_ids.append(torch.tensor(pos, dtype=torch.long))
            self.label_ids.append(torch.tensor(labels, dtype=torch.long))
            self.lengths.append(len(sent))

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        return (self.word_ids[idx], self.pos_ids[idx],
                self.label_ids[idx], self.lengths[idx])


def collate_fn(batch):
    """Custom collate function for padding variable-length sequences."""
    words, pos, labels, lengths = zip(*batch)

    # Sort by length (descending) for pack_padded_sequence
    sorted_indices = sorted(range(len(lengths)),
                           key=lambda i: lengths[i], reverse=True)

    words = pad_sequence([words[i] for i in sorted_indices], batch_first=True)
    pos = pad_sequence([pos[i] for i in sorted_indices], batch_first=True)
    labels = pad_sequence([labels[i] for i in sorted_indices],
                         batch_first=True, padding_value=-1)
    lengths = [lengths[i] for i in sorted_indices]

    return words, pos, labels, lengths, sorted_indices


# ==================== Model ====================

class BiLSTMClauseDetector(nn.Module):
    """Bidirectional LSTM model for clause boundary detection."""

    def __init__(self, word_vocab_size: int, pos_vocab_size: int,
                 num_labels: int, word_emb_dim: int = 100,
                 pos_emb_dim: int = 25, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            word_vocab_size: Size of word vocabulary
            pos_vocab_size: Size of POS tag vocabulary
            num_labels: Number of output labels (B-CLAUSE, I-CLAUSE, O)
            word_emb_dim: Dimension of word embeddings
            pos_emb_dim: Dimension of POS tag embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=word_emb_dim + pos_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)  # *2 for bidirectional

    def forward(self, words, pos, lengths):
        """
        Forward pass.

        Args:
            words: Word indices [batch_size, seq_len]
            pos: POS indices [batch_size, seq_len]
            lengths: Actual lengths of sequences

        Returns:
            Logits [batch_size, seq_len, num_labels]
        """
        word_emb = self.word_embedding(words)
        pos_emb = self.pos_embedding(pos)
        x = torch.cat([word_emb, pos_emb], dim=-1)  # [batch, seq, emb_dim]

        x = self.dropout(x)

        # Pack for efficient computation
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                      enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # [batch, seq, num_labels]

        return logits


# ==================== Trainer ====================

class BiLSTMTrainer:
    """Handles training, evaluation, and prediction for the BiLSTM model."""

    def __init__(self, word_emb_dim: int = 100, pos_emb_dim: int = 25,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, lr: float = 0.001,
                 batch_size: int = 32, device: str = None):
        """
        Initialize the trainer.

        Args:
            word_emb_dim: Word embedding dimension
            pos_emb_dim: POS embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Batch size
            device: 'cuda' or 'cpu'
        """
        self.word_emb_dim = word_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.word_vocab = Vocabulary()
        self.pos_vocab = Vocabulary()
        self.label_vocab = Vocabulary(pad_token="<PAD>", unk_token="O")
        self.model = None
        self.is_trained = False

    def build_vocabs(self, train_data: List[List[Tuple]]):
        """Build vocabularies from training data."""
        words = []
        pos_tags = []
        labels = []

        for sent in train_data:
            for token in sent:
                words.append(token[0].lower())
                pos_tags.append(token[1])
                labels.append(token[-1])

        self.word_vocab.build_from_data(words, min_freq=2)
        self.pos_vocab.build_from_data(pos_tags)

        # Label vocab (no min_freq filter)
        for label in set(labels):
            self.label_vocab.add_token(label)

        print(f"Vocabularies built:")
        print(f"  Words: {len(self.word_vocab)}")
        print(f"  POS tags: {len(self.pos_vocab)}")
        print(f"  Labels: {len(self.label_vocab)}")

    def train(self, train_data: List[List[Tuple]],
              dev_data: Optional[List[List[Tuple]]] = None,
              epochs: int = 20, patience: int = 5) -> Dict[str, Any]:
        """
        Train the BiLSTM model.

        Args:
            train_data: Training data
            dev_data: Dev data for early stopping
            epochs: Maximum training epochs
            patience: Early stopping patience

        Returns:
            Training history dict
        """
        print(f"Training on device: {self.device}")

        # Build vocabularies
        self.build_vocabs(train_data)

        # Create datasets
        train_dataset = ClauseDataset(train_data, self.word_vocab,
                                       self.pos_vocab, self.label_vocab)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                   shuffle=True, collate_fn=collate_fn)

        dev_loader = None
        if dev_data:
            dev_dataset = ClauseDataset(dev_data, self.word_vocab,
                                         self.pos_vocab, self.label_vocab)
            dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size,
                                     shuffle=False, collate_fn=collate_fn)

        # Initialize model
        self.model = BiLSTMClauseDetector(
            word_vocab_size=len(self.word_vocab),
            pos_vocab_size=len(self.pos_vocab),
            num_labels=len(self.label_vocab),
            word_emb_dim=self.word_emb_dim,
            pos_emb_dim=self.pos_emb_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Compute class weights for imbalanced data
        label_counts = Counter()
        for sent in train_data:
            for t in sent:
                label_counts[t[-1]] += 1

        total = sum(label_counts.values())
        weights = []
        for i in range(len(self.label_vocab)):
            label = self.label_vocab.decode(i)
            count = label_counts.get(label, 1)
            weights.append(total / (len(label_counts) * count))
        class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )

        # Training loop
        history = {"train_loss": [], "dev_f1": []}
        best_f1 = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for words, pos, labels, lengths, _ in progress:
                words = words.to(self.device)
                pos = pos.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(words, pos, lengths)

                # Reshape for loss
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)

                loss = criterion(logits, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / num_batches
            history["train_loss"].append(avg_loss)

            # Evaluate on dev set
            if dev_loader:
                dev_f1 = self._evaluate_f1(dev_loader)
                history["dev_f1"].append(dev_f1)
                scheduler.step(dev_f1)

                print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, dev_F1={dev_f1:.4f}")

                # Early stopping
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in
                                  self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.is_trained = True
        history["best_dev_f1"] = best_f1

        return history

    def _evaluate_f1(self, data_loader: DataLoader) -> float:
        """Compute F1 score on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for words, pos, labels, lengths, _ in data_loader:
                words = words.to(self.device)
                pos = pos.to(self.device)

                logits = self.model(words, pos, lengths)
                preds = torch.argmax(logits, dim=-1)

                for i in range(len(lengths)):
                    length = lengths[i]
                    pred_seq = preds[i][:length].cpu().tolist()
                    label_seq = labels[i][:length].tolist()

                    all_preds.extend(pred_seq)
                    all_labels.extend(label_seq)

        # Compute F1 for B-CLAUSE and I-CLAUSE
        from sklearn.metrics import f1_score
        b_idx = self.label_vocab.encode("B-CLAUSE")
        i_idx = self.label_vocab.encode("I-CLAUSE")
        labels_to_eval = [b_idx, i_idx]

        # Filter valid entries
        valid = [(p, l) for p, l in zip(all_preds, all_labels) if l != -1]
        if not valid:
            return 0.0

        preds, labels = zip(*valid)
        return f1_score(labels, preds, labels=labels_to_eval, average="macro",
                       zero_division=0)

    def predict(self, data: List[List[Tuple]]) -> List[List[str]]:
        """
        Predict BIO labels for sentences.

        Args:
            data: List of sentences (same format as training, labels ignored)

        Returns:
            List of predicted label sequences
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        dataset = ClauseDataset(data, self.word_vocab,
                                 self.pos_vocab, self.label_vocab)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                             shuffle=False, collate_fn=collate_fn)

        self.model.eval()
        all_predictions = [None] * len(data)

        global_offset = 0
        with torch.no_grad():
            for words, pos, labels, lengths, sorted_indices in loader:
                batch_size_actual = len(sorted_indices)
                words = words.to(self.device)
                pos = pos.to(self.device)

                logits = self.model(words, pos, lengths)
                preds = torch.argmax(logits, dim=-1)

                for i, batch_local_idx in enumerate(sorted_indices):
                    # Map batch-local index back to global dataset index
                    global_idx = global_offset + batch_local_idx
                    length = lengths[i]
                    pred_labels = [
                        self.label_vocab.decode(idx)
                        for idx in preds[i][:length].cpu().tolist()
                    ]
                    all_predictions[global_idx] = pred_labels

                global_offset += batch_size_actual

        return all_predictions

    def evaluate(self, test_data: List[List[Tuple]]) -> Dict[str, Any]:
        """
        Full evaluation on test data.

        Args:
            test_data: Test data

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(test_data)
        gold_labels = [[t[-1] for t in sent] for sent in test_data]

        from sklearn.metrics import classification_report

        y_true_flat = [l for sent in gold_labels for l in sent]
        y_pred_flat = [l for sent in predictions for l in sent]

        report = classification_report(
            y_true_flat, y_pred_flat,
            labels=["B-CLAUSE", "I-CLAUSE", "O"],
            output_dict=True,
        )

        return {
            "overall": report,
            "y_test": gold_labels,
            "y_pred": predictions,
        }

    def save(self, model_dir: str):
        """Save model, vocabularies, and config."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), model_dir / "bilstm_weights.pth")

        # Save vocabularies
        self.word_vocab.save(str(model_dir / "word_vocab.json"))
        self.pos_vocab.save(str(model_dir / "pos_vocab.json"))
        self.label_vocab.save(str(model_dir / "label_vocab.json"))

        # Save config
        config = {
            "word_emb_dim": self.word_emb_dim,
            "pos_emb_dim": self.pos_emb_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "word_vocab_size": len(self.word_vocab),
            "pos_vocab_size": len(self.pos_vocab),
            "num_labels": len(self.label_vocab),
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {model_dir}")

    def load(self, model_dir: str):
        """Load model, vocabularies, and config."""
        model_dir = Path(model_dir)

        # Load config
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)

        # Load vocabularies
        self.word_vocab.load(str(model_dir / "word_vocab.json"))
        self.pos_vocab.load(str(model_dir / "pos_vocab.json"))
        self.label_vocab.load(str(model_dir / "label_vocab.json"))

        # Rebuild model
        self.model = BiLSTMClauseDetector(
            word_vocab_size=config["word_vocab_size"],
            pos_vocab_size=config["pos_vocab_size"],
            num_labels=config["num_labels"],
            word_emb_dim=config["word_emb_dim"],
            pos_emb_dim=config["pos_emb_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_dir / "bilstm_weights.pth",
                       map_location=self.device)
        )
        self.is_trained = True
        print(f"Model loaded from {model_dir}")


if __name__ == "__main__":
    # Quick test
    sample = [
        [
            ("When", "SCONJ", "mark", "WRB", 2, "B-CLAUSE"),
            ("rain", "NOUN", "nsubj", "NN", 1, "I-CLAUSE"),
            ("stopped", "VERB", "advcl", "VBD", 3, "I-CLAUSE"),
            ("we", "PRON", "nsubj", "PRP", 1, "B-CLAUSE"),
            ("went", "VERB", "root", "VBD", 0, "I-CLAUSE"),
            ("outside", "ADV", "advmod", "RB", 1, "I-CLAUSE"),
        ]
    ] * 50

    trainer = BiLSTMTrainer(hidden_dim=64, batch_size=16, device="cpu")
    history = trainer.train(sample, dev_data=sample[:10], epochs=5)
    print(f"Best dev F1: {history['best_dev_f1']:.4f}")
