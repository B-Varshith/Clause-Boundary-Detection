"""
clause_labeler.py - Generate ground-truth BIO clause boundary labels from UD dependency trees.

Uses dependency relations to identify clause boundaries and assigns BIO tags
(B-CLAUSE, I-CLAUSE, O) to each token in a sentence.
"""

from typing import List, Dict, Tuple, Set, Any
from src.data_loader import extract_tokens


# Dependency relations that indicate a clause boundary (subordinate/embedded clause)
CLAUSE_DEPRELS = {
    "advcl",      # Adverbial clause modifier ("When the rain stopped, ...")
    "ccomp",      # Clausal complement ("She said that he will come")
    "xcomp",      # Open clausal complement ("I want to go")
    "acl",        # Adnominal clause ("the man sitting there")
    "acl:relcl",  # Relative clause ("the book that I read")
    "parataxis",  # Paratactic clause ("He said: let's go")
    "csubj",      # Clausal subject ("What he did was wrong")
    "csubj:pass", # Passive clausal subject
}

# Relations that indicate coordination of clauses
COORD_DEPRELS = {"conj"}


def build_children_map(tokens: List[Dict]) -> Dict[int, List[int]]:
    """
    Build a mapping from head token ID to list of child token IDs.

    Args:
        tokens: List of token dictionaries

    Returns:
        Dict mapping head_id -> [child_id_1, child_id_2, ...]
    """
    children = {}
    for token in tokens:
        head = token["head"]
        tid = token["id"]
        if head not in children:
            children[head] = []
        children[head].append(tid)
    return children


def get_subtree_ids(token_id: int, children_map: Dict[int, List[int]],
                     exclude_ids: Set[int] = None) -> Set[int]:
    """
    Get all token IDs in the subtree rooted at token_id.
    Optionally excludes tokens belonging to specified subtrees.

    Args:
        token_id: Root of the subtree
        children_map: Mapping from head to children
        exclude_ids: Set of token IDs to exclude (nested clause heads)

    Returns:
        Set of token IDs in the subtree
    """
    if exclude_ids is None:
        exclude_ids = set()

    result = {token_id}
    if token_id in children_map:
        for child_id in children_map[token_id]:
            if child_id not in exclude_ids:
                result.update(get_subtree_ids(child_id, children_map, exclude_ids))
    return result


def identify_clauses(tokens: List[Dict]) -> List[Dict[str, Any]]:
    """
    Identify clauses in a sentence based on dependency relations.

    Strategy:
    1. Find all clause-indicating dependency relations
    2. Each such token heads a subordinate/embedded clause
    3. The root verb heads the main clause
    4. Handle coordinated verbs via 'conj' relation

    Args:
        tokens: List of token dictionaries

    Returns:
        List of clause dicts with 'head_id', 'type', 'token_ids' keys
    """
    if not tokens:
        return []

    children_map = build_children_map(tokens)
    token_map = {t["id"]: t for t in tokens}

    # Step 1: Find the root token
    root_id = None
    for token in tokens:
        if token["deprel"] == "root":
            root_id = token["id"]
            break

    if root_id is None:
        # Fallback: treat the entire sentence as one clause
        return [{
            "head_id": tokens[0]["id"],
            "type": "root",
            "token_ids": sorted([t["id"] for t in tokens])
        }]

    # Step 2: Find all clause heads (tokens with clause-indicating deprels)
    clause_heads = []
    for token in tokens:
        if token["deprel"] in CLAUSE_DEPRELS:
            clause_heads.append({
                "head_id": token["id"],
                "type": token["deprel"],
            })
        # Handle coordinated verbs that form separate clauses
        elif (token["deprel"] in COORD_DEPRELS and
              token.get("upos") in ("VERB", "AUX") and
              token_map.get(token["head"], {}).get("upos") in ("VERB", "AUX")):
            clause_heads.append({
                "head_id": token["id"],
                "type": "conj_clause",
            })

    # Step 3: Collect subtree token IDs for each clause
    # Subordinate clause heads to exclude from parent clauses
    sub_clause_head_ids = {ch["head_id"] for ch in clause_heads}

    clauses = []
    for ch in clause_heads:
        # Get subtree but exclude nested sub-clause heads within this clause
        nested_heads = sub_clause_head_ids - {ch["head_id"]}
        token_ids = get_subtree_ids(ch["head_id"], children_map, nested_heads)
        clauses.append({
            "head_id": ch["head_id"],
            "type": ch["type"],
            "token_ids": sorted(token_ids),
        })

    # Step 4: Main clause = root subtree minus all subordinate clause tokens
    all_sub_ids = set()
    for clause in clauses:
        all_sub_ids.update(clause["token_ids"])

    main_clause_ids = get_subtree_ids(root_id, children_map, sub_clause_head_ids)
    # Remove any IDs already claimed by subordinate clauses
    main_clause_ids -= all_sub_ids
    # Always include the root itself in the main clause
    main_clause_ids.add(root_id)

    main_clause = {
        "head_id": root_id,
        "type": "root",
        "token_ids": sorted(main_clause_ids),
    }

    # Combine: main clause first, then subordinate clauses
    all_clauses = [main_clause] + clauses

    return all_clauses


def generate_bio_labels(sentence) -> List[Tuple[str, str, str]]:
    """
    Generate BIO labels for each token in a sentence.

    Args:
        sentence: A parsed sentence from conllu library

    Returns:
        List of tuples (word, pos, bio_label) for each token
    """
    tokens = extract_tokens(sentence)
    if not tokens:
        return []

    clauses = identify_clauses(tokens)

    # Assign BIO labels
    token_labels = {}
    for clause in clauses:
        if not clause["token_ids"]:
            continue
        first_id = clause["token_ids"][0]
        for tid in clause["token_ids"]:
            if tid not in token_labels:  # First assignment wins
                if tid == first_id:
                    token_labels[tid] = "B-CLAUSE"
                else:
                    token_labels[tid] = "I-CLAUSE"

    # Build result
    result = []
    for token in tokens:
        label = token_labels.get(token["id"], "O")
        result.append((token["form"], token["upos"], label))

    return result


def generate_labels_for_dataset(sentences) -> List[List[Tuple[str, str, str]]]:
    """
    Generate BIO labels for an entire dataset split.

    Args:
        sentences: List of parsed sentences

    Returns:
        List of labeled sentences, each being a list of (word, pos, label) tuples
    """
    labeled = []
    for sentence in sentences:
        bio = generate_bio_labels(sentence)
        if bio:
            labeled.append(bio)
    return labeled


def get_clause_spans(bio_labels: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    Convert BIO labels back to clause spans for visualization.

    Args:
        bio_labels: List of (word, pos, label) tuples

    Returns:
        List of clause dicts with 'text', 'start', 'end' keys
    """
    clauses = []
    current_clause = []
    start_idx = 0

    for i, (word, pos, label) in enumerate(bio_labels):
        if label == "B-CLAUSE":
            if current_clause:
                clauses.append({
                    "text": " ".join(current_clause),
                    "start": start_idx,
                    "end": i - 1,
                })
            current_clause = [word]
            start_idx = i
        elif label == "I-CLAUSE":
            current_clause.append(word)
        else:  # O
            if current_clause:
                clauses.append({
                    "text": " ".join(current_clause),
                    "start": start_idx,
                    "end": i - 1,
                })
                current_clause = []

    # Don't forget the last clause
    if current_clause:
        clauses.append({
            "text": " ".join(current_clause),
            "start": start_idx,
            "end": len(bio_labels) - 1,
        })

    return clauses


if __name__ == "__main__":
    from src.data_loader import load_conllu, get_sentence_text

    # Quick test with sample data
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/UD_English-EWT/en_ewt-ud-dev.conllu"

    try:
        sentences = load_conllu(filepath)
        print(f"Loaded {len(sentences)} sentences\n")

        # Show first 5 sentences with clause labels
        for i, sent in enumerate(sentences[:5]):
            text = get_sentence_text(sent)
            bio = generate_bio_labels(sent)
            clauses = get_clause_spans(bio)

            print(f"Sentence {i+1}: {text}")
            print(f"  Clauses ({len(clauses)}):")
            for j, c in enumerate(clauses):
                print(f"    [{j+1}] {c['text']}")
            print(f"  BIO tags: {[l for _, _, l in bio]}")
            print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the UD_English-EWT dataset first.")
