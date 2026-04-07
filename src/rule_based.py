"""
rule_based.py - Rule-based clause boundary detector using spaCy dependency parsing.

Identifies clause boundaries using linguistic rules based on dependency relations,
subordinating conjunctions, and verb-argument structure.
"""

import spacy
from typing import List, Dict, Tuple, Any

# Load spaCy model (will be downloaded during setup)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy English model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Dependency relations that indicate a clause boundary
CLAUSE_RELS = {
    "advcl",      # Adverbial clause
    "ccomp",      # Clausal complement
    "xcomp",      # Open clausal complement
    "acl",        # Adnominal clause
    "relcl",      # Relative clause
    "parataxis",  # Paratactic
    "csubj",      # Clausal subject
    "csubjpass",  # Passive clausal subject
}


def get_subtree_tokens(token, exclude_heads=None):
    """
    Get all tokens in the subtree of a given token,
    excluding tokens belonging to nested clause subtrees.

    Args:
        token: spaCy Token
        exclude_heads: Set of token indices whose subtrees should be excluded

    Returns:
        List of spaCy Tokens in the subtree
    """
    if exclude_heads is None:
        exclude_heads = set()

    result = [token]
    for child in token.children:
        if child.i not in exclude_heads:
            result.extend(get_subtree_tokens(child, exclude_heads))
    return result


def detect_clauses_spacy(text: str) -> List[Dict[str, Any]]:
    """
    Detect clause boundaries in a text using spaCy dependency parsing.

    Args:
        text: Input text string

    Returns:
        List of clause dicts with 'text', 'type', 'tokens', 'start', 'end' keys
    """
    doc = nlp(text)
    all_clauses = []

    for sent in doc.sents:
        clauses = _detect_clauses_in_sentence(sent)
        all_clauses.extend(clauses)

    return all_clauses


def _detect_clauses_in_sentence(sent) -> List[Dict[str, Any]]:
    """
    Detect clauses within a single sentence span.
    """
    tokens = list(sent)
    if not tokens:
        return []

    # Find root
    root = None
    for token in tokens:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None:
        return [{
            "text": sent.text,
            "type": "root",
            "tokens": [t.text for t in tokens],
            "start": sent.start,
            "end": sent.end,
        }]

    # Find all clause heads
    clause_heads = []
    for token in tokens:
        if token.dep_ in CLAUSE_RELS:
            clause_heads.append(token)
        # Coordinated verbs forming separate clauses
        elif (token.dep_ == "conj" and
              token.pos_ in ("VERB", "AUX") and
              token.head.pos_ in ("VERB", "AUX")):
            clause_heads.append(token)

    # Build clause subtrees
    clause_head_indices = {ch.i for ch in clause_heads}
    clauses = []

    for ch in clause_heads:
        # Get subtree but exclude nested clause heads
        nested = clause_head_indices - {ch.i}
        subtree_tokens = get_subtree_tokens(ch, nested)
        subtree_tokens.sort(key=lambda t: t.i)

        clauses.append({
            "text": " ".join(t.text for t in subtree_tokens),
            "type": ch.dep_,
            "tokens": [t.text for t in subtree_tokens],
            "start": subtree_tokens[0].i,
            "end": subtree_tokens[-1].i,
            "head_token": ch.text,
        })

    # Main clause: root subtree minus subordinate clause tokens
    sub_token_indices = set()
    for clause in clauses:
        sub_token_indices.update(t.i for t in get_subtree_tokens(
            [t for t in tokens if t.i == clause["start"]][0] if clause["tokens"] else root,
            clause_head_indices - {[ch for ch in clause_heads if ch.text == clause["head_token"]][0].i}
            if clause["head_token"] else set()
        ))

    # Simpler approach: main clause = all tokens not in any subordinate clause
    all_sub_indices = set()
    for ch in clause_heads:
        for t in ch.subtree:
            all_sub_indices.add(t.i)

    main_tokens = [t for t in tokens if t.i not in all_sub_indices]
    if not main_tokens:
        main_tokens = [root]

    main_clause = {
        "text": " ".join(t.text for t in main_tokens),
        "type": "root",
        "tokens": [t.text for t in main_tokens],
        "start": main_tokens[0].i if main_tokens else sent.start,
        "end": main_tokens[-1].i if main_tokens else sent.end,
        "head_token": root.text,
    }

    # Combine and sort by position
    all_clauses = [main_clause] + clauses
    all_clauses.sort(key=lambda c: c["start"])

    return all_clauses


def predict_bio_tags(text: str) -> List[Tuple[str, str]]:
    """
    Predict BIO tags for each token in the input text.

    Args:
        text: Input text string

    Returns:
        List of (word, predicted_bio_tag) tuples
    """
    doc = nlp(text)
    results = []

    for sent in doc.sents:
        tokens = list(sent)
        clauses = _detect_clauses_in_sentence(sent)

        # Map token index to clause
        token_to_clause = {}
        for clause in clauses:
            clause_start = clause["start"]
            for j, tok_text in enumerate(clause["tokens"]):
                idx = clause_start + j
                if idx not in token_to_clause:
                    token_to_clause[idx] = clause

        # Assign BIO tags
        for token in tokens:
            if token.i in token_to_clause:
                clause = token_to_clause[token.i]
                if token.i == clause["start"]:
                    results.append((token.text, "B-CLAUSE"))
                else:
                    results.append((token.text, "I-CLAUSE"))
            else:
                results.append((token.text, "O"))

    return results


def predict_bio_for_tokens(tokens: List[str]) -> List[str]:
    """
    Predict BIO tags given a list of tokens (reconstructs text and parses).

    Args:
        tokens: List of word strings

    Returns:
        List of BIO labels
    """
    text = " ".join(tokens)
    doc = nlp(text)

    # Get spaCy's tokenization
    spacy_bio = predict_bio_tags(text)

    # Align with input tokens (simple approach)
    if len(spacy_bio) == len(tokens):
        return [label for _, label in spacy_bio]

    # If tokenization differs, fall back to simple alignment
    labels = []
    spacy_idx = 0
    for tok in tokens:
        if spacy_idx < len(spacy_bio):
            labels.append(spacy_bio[spacy_idx][1])
            spacy_idx += 1
        else:
            labels.append("O")

    return labels


if __name__ == "__main__":
    # Quick test
    test_sentences = [
        "When the rain stopped, we went outside and played.",
        "She said that he will come tomorrow.",
        "The book that I read last week was amazing.",
        "If you study well, you will succeed.",
        "I think that he knows that she left.",
    ]

    for text in test_sentences:
        print(f"Sentence: {text}")
        clauses = detect_clauses_spacy(text)
        print(f"  Clauses ({len(clauses)}):")
        for i, c in enumerate(clauses):
            print(f"    [{i+1}] ({c['type']}): {c['text']}")

        bio = predict_bio_tags(text)
        print(f"  BIO: {[(w, l) for w, l in bio]}")
        print()
