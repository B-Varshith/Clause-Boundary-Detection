"""
data_loader.py - Parse CoNLL-U files from Universal Dependencies dataset.

Loads .conllu files and extracts structured token information including
words, POS tags, dependency relations, and sentence metadata.
"""

from conllu import parse
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_conllu(filepath: str) -> List[List[Dict[str, Any]]]:
    """
    Parse a .conllu file and return a list of sentences.
    Each sentence is a list of token dictionaries.

    Args:
        filepath: Path to the .conllu file

    Returns:
        List of sentences, each sentence is a list of token dicts with keys:
        id, form, lemma, upos, xpos, feats, head, deprel, deps, misc
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()

    sentences = parse(data)
    return sentences


def load_dataset(data_dir: str) -> Dict[str, List]:
    """
    Load train, dev, and test splits from a UD treebank directory.

    Args:
        data_dir: Path to the UD treebank directory (e.g., data/UD_English-EWT/)

    Returns:
        Dictionary with 'train', 'dev', 'test' keys mapping to lists of sentences
    """
    data_dir = Path(data_dir)
    dataset = {}

    for split in ["train", "dev", "test"]:
        # UD files follow pattern: <lang>_<treebank>-ud-<split>.conllu
        pattern = f"*-ud-{split}.conllu"
        files = list(data_dir.glob(pattern))
        if not files:
            print(f"Warning: No {split} file found matching {pattern} in {data_dir}")
            dataset[split] = []
            continue
        filepath = files[0]
        print(f"Loading {split}: {filepath.name}")
        dataset[split] = load_conllu(str(filepath))

    return dataset


def extract_tokens(sentence) -> List[Dict[str, Any]]:
    """
    Extract clean token information from a parsed sentence.
    Skips multi-word tokens (IDs like '1-2') and empty nodes.

    Args:
        sentence: A parsed sentence from conllu library

    Returns:
        List of token dicts with cleaned fields
    """
    tokens = []
    for token in sentence:
        # Skip multi-word tokens and empty nodes
        if isinstance(token["id"], tuple) or isinstance(token["id"], list):
            continue
        if token["id"] is None:
            continue

        tokens.append({
            "id": token["id"],
            "form": token["form"],
            "lemma": token.get("lemma", "_"),
            "upos": token.get("upos", "_"),
            "xpos": token.get("xpos", "_"),
            "feats": token.get("feats", None),
            "head": token.get("head", 0),
            "deprel": token.get("deprel", "_"),
            "deps": token.get("deps", None),
            "misc": token.get("misc", None),
        })
    return tokens


def get_sentence_text(sentence) -> str:
    """Extract the original text from a sentence's metadata."""
    if hasattr(sentence, "metadata") and "text" in sentence.metadata:
        return sentence.metadata["text"]
    # Fallback: reconstruct from tokens
    tokens = extract_tokens(sentence)
    return " ".join(t["form"] for t in tokens)


def get_dataset_stats(dataset: Dict[str, List]) -> Dict[str, Any]:
    """
    Compute basic statistics about the dataset.

    Returns:
        Dictionary with stats for each split
    """
    stats = {}
    for split, sentences in dataset.items():
        if not sentences:
            stats[split] = {"num_sentences": 0, "num_tokens": 0}
            continue

        num_tokens = sum(len(extract_tokens(s)) for s in sentences)
        avg_len = num_tokens / len(sentences) if sentences else 0

        stats[split] = {
            "num_sentences": len(sentences),
            "num_tokens": num_tokens,
            "avg_sentence_length": round(avg_len, 2),
        }
    return stats


if __name__ == "__main__":
    # Quick test
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/UD_English-EWT"
    dataset = load_dataset(data_dir)
    stats = get_dataset_stats(dataset)

    print("\n=== Dataset Statistics ===")
    for split, s in stats.items():
        print(f"  {split}: {s}")
