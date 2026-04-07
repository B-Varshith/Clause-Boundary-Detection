"""
feature_extractor.py - Feature engineering for CRF sequence labeling model.

Extracts rich linguistic features from tokens for clause boundary detection,
including POS tags, dependency relations, morphological features, and context windows.
"""

from typing import List, Dict, Any, Tuple


def word2features(sent: List[Tuple[str, str, str, str, int]],
                  i: int) -> Dict[str, Any]:
    """
    Extract features for a single token at position i in a sentence.

    Args:
        sent: List of tuples (word, upos, deprel, xpos, head_distance)
        i: Token index

    Returns:
        Dictionary of feature name -> feature value
    """
    word = sent[i][0]
    upos = sent[i][1]
    deprel = sent[i][2]
    xpos = sent[i][3]
    head_dist = sent[i][4]

    features = {
        "bias": 1.0,
        # Word features
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.length": len(word),
        # POS features
        "upos": upos,
        "xpos": xpos,
        # Dependency features
        "deprel": deprel,
        "head_distance": head_dist,
        # Linguistic indicators
        "is_verb": upos in ("VERB", "AUX"),
        "is_sconj": upos == "SCONJ",
        "is_cconj": upos == "CCONJ",
        "is_punct": upos == "PUNCT",
        "is_pron": upos == "PRON",
        # Clause-indicating deprels
        "is_clause_deprel": deprel in (
            "advcl", "ccomp", "xcomp", "acl", "acl:relcl",
            "parataxis", "csubj", "csubj:pass"
        ),
        "is_root": deprel == "root",
        "is_conj": deprel == "conj",
        "is_mark": deprel == "mark",
        # Subordinating markers
        "is_subordinator": word.lower() in (
            "when", "while", "if", "because", "since", "although",
            "though", "unless", "until", "after", "before", "that",
            "which", "who", "whom", "whose", "where", "whereas"
        ),
        # Comma (often clause boundary)
        "is_comma": word == ",",
    }

    # Context: Previous token features
    if i > 0:
        prev_word = sent[i-1][0]
        prev_upos = sent[i-1][1]
        prev_deprel = sent[i-1][2]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:upos": prev_upos,
            "-1:deprel": prev_deprel,
            "-1:is_verb": prev_upos in ("VERB", "AUX"),
            "-1:is_punct": prev_upos == "PUNCT",
            "-1:is_comma": prev_word == ",",
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    # Context: Two tokens back
    if i > 1:
        features.update({
            "-2:word.lower()": sent[i-2][0].lower(),
            "-2:upos": sent[i-2][1],
        })

    # Context: Next token features
    if i < len(sent) - 1:
        next_word = sent[i+1][0]
        next_upos = sent[i+1][1]
        next_deprel = sent[i+1][2]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:upos": next_upos,
            "+1:deprel": next_deprel,
            "+1:is_verb": next_upos in ("VERB", "AUX"),
            "+1:is_sconj": next_upos == "SCONJ",
            "+1:is_comma": next_word == ",",
        })
    else:
        features["EOS"] = True  # End of sentence

    # Context: Two tokens ahead
    if i < len(sent) - 2:
        features.update({
            "+2:word.lower()": sent[i+2][0].lower(),
            "+2:upos": sent[i+2][1],
        })

    return features


def sent2features(sent: List[Tuple]) -> List[Dict[str, Any]]:
    """
    Extract features for all tokens in a sentence.

    Args:
        sent: List of token tuples

    Returns:
        List of feature dictionaries
    """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent: List[Tuple]) -> List[str]:
    """
    Extract BIO labels for all tokens in a sentence.

    Args:
        sent: List of tuples where the last element is the label

    Returns:
        List of BIO labels
    """
    return [s[-1] for s in sent]


def sent2tokens(sent: List[Tuple]) -> List[str]:
    """
    Extract word tokens from a sentence.

    Args:
        sent: List of tuples where the first element is the word

    Returns:
        List of word strings
    """
    return [s[0] for s in sent]


def prepare_crf_data(labeled_sentences, tokens_data) -> List[List[Tuple]]:
    """
    Prepare data in the format needed for CRF feature extraction.
    Combines token info with BIO labels.

    Args:
        labeled_sentences: List of [(word, pos, label), ...] from clause_labeler
        tokens_data: List of sentence token dicts from data_loader

    Returns:
        List of sentences, each being a list of
        (word, upos, deprel, xpos, head_distance, label) tuples
    """
    prepared = []
    for labeled, sent_tokens in zip(labeled_sentences, tokens_data):
        sent_data = []
        for (word, upos, label), token in zip(labeled, sent_tokens):
            head_dist = abs(token["id"] - token["head"]) if token["head"] else 0
            sent_data.append((
                word,
                upos,
                token.get("deprel", "_"),
                token.get("xpos", "_"),
                head_dist,
                label,
            ))
        prepared.append(sent_data)
    return prepared


if __name__ == "__main__":
    # Quick test with dummy data
    sample = [
        ("When", "SCONJ", "mark", "WRB", 2, "B-CLAUSE"),
        ("the", "DET", "det", "DT", 1, "I-CLAUSE"),
        ("rain", "NOUN", "nsubj", "NN", 1, "I-CLAUSE"),
        ("stopped", "VERB", "advcl", "VBD", 3, "I-CLAUSE"),
        (",", "PUNCT", "punct", ",", 3, "O"),
        ("we", "PRON", "nsubj", "PRP", 1, "B-CLAUSE"),
        ("went", "VERB", "root", "VBD", 0, "I-CLAUSE"),
        ("outside", "ADV", "advmod", "RB", 1, "I-CLAUSE"),
    ]

    features = sent2features(sample)
    labels = sent2labels(sample)

    print("Sample features for 'When':")
    for k, v in features[0].items():
        print(f"  {k}: {v}")

    print(f"\nLabels: {labels}")
