import re
import math
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple


def tokenize(text: str) -> List[str]:
    """Lowercase and clean string into word tokens."""
    if not text:
        return []
    # Replace punctuation with spaces and split
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if w]


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Generate n-gram frequency counter."""
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])


def calculate_bleu(candidate: str, reference: str, max_n: int = 4) -> Dict[str, Any]:
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 and cumulative BLEU score.
    Follows Papineni et al. (2002) with modified n-gram precision and Brevity Penalty.
    """
    c_tokens = tokenize(candidate)
    r_tokens = tokenize(reference)

    c_len = len(c_tokens)
    r_len = len(r_tokens)

    if c_len == 0 or r_len == 0:
        return {
            "bleu_1": 0.0,
            "bleu_2": 0.0,
            "bleu_3": 0.0,
            "bleu_4": 0.0,
            "cumulative_bleu": 0.0,
            "brevity_penalty": 0.0
        }

    # 1. Brevity Penalty
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (float(r_len) / float(c_len)))

    precisions = []
    bleu_scores = {}

    for n in range(1, max_n + 1):
        c_ngrams = get_ngrams(c_tokens, n)
        r_ngrams = get_ngrams(r_tokens, n)

        total_candidate_ngrams = sum(c_ngrams.values())
        if total_candidate_ngrams == 0:
            p_n = 0.0
        else:
            # Clipped count
            clipped_count = sum(min(count, r_ngrams.get(ngram, 0)) for ngram, count in c_ngrams.items())
            p_n = clipped_count / total_candidate_ngrams

        precisions.append(p_n)
        bleu_scores[f"bleu_{n}"] = round(p_n * 100.0, 2)

    # Cumulative BLEU-4 with uniform weights [0.25, 0.25, 0.25, 0.25]
    # Use smoothing for higher-order n-grams if zero
    log_precision_sum = 0.0
    weight = 1.0 / max_n

    for p_n in precisions:
        if p_n > 0:
            log_precision_sum += weight * math.log(p_n)
        else:
            # Additive Laplace smoothing for log of zero precision
            log_precision_sum += weight * math.log(1e-4)

    cumulative_bleu = bp * math.exp(log_precision_sum)
    bleu_scores["cumulative_bleu"] = round(min(max(cumulative_bleu * 100.0, 0.0), 100.0), 2)
    bleu_scores["brevity_penalty"] = round(bp, 3)

    return bleu_scores


def calculate_lcs(x: List[str], y: List[str]) -> int:
    """Compute Longest Common Subsequence length between two token lists."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def calculate_rouge(candidate: str, reference: str) -> Dict[str, Any]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L (F1, Precision, Recall).
    Follows Lin (2004).
    """
    c_tokens = tokenize(candidate)
    r_tokens = tokenize(reference)

    c_len = len(c_tokens)
    r_len = len(r_tokens)

    if c_len == 0 or r_len == 0:
        return {
            "rouge_1": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "rouge_2": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "rouge_l": {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        }

    # Helper for n-gram ROUGE
    def n_gram_rouge(n: int):
        c_ng = get_ngrams(c_tokens, n)
        r_ng = get_ngrams(r_tokens, n)
        overlap = sum(min(count, r_ng.get(ng, 0)) for ng, count in c_ng.items())
        total_cand = sum(c_ng.values())
        total_ref = sum(r_ng.values())

        p = (overlap / total_cand) if total_cand > 0 else 0.0
        r = (overlap / total_ref) if total_ref > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return {
            "f1": round(f1 * 100.0, 2),
            "precision": round(p * 100.0, 2),
            "recall": round(r * 100.0, 2)
        }

    # ROUGE-L using LCS
    lcs_len = calculate_lcs(c_tokens, r_tokens)
    rl_p = lcs_len / c_len
    rl_r = lcs_len / r_len
    rl_f1 = (2 * rl_p * rl_r / (rl_p + rl_r)) if (rl_p + rl_r) > 0 else 0.0

    return {
        "rouge_1": n_gram_rouge(1),
        "rouge_2": n_gram_rouge(2),
        "rouge_l": {
            "f1": round(rl_f1 * 100.0, 2),
            "precision": round(rl_p * 100.0, 2),
            "recall": round(rl_r * 100.0, 2)
        }
    }


def calculate_meteor(candidate: str, reference: str) -> Dict[str, Any]:
    """
    Calculate METEOR score (Banerjee and Lavie, 2005) with exact unigram matches
    and chunk fragmentation penalty.
    """
    c_tokens = tokenize(candidate)
    r_tokens = tokenize(reference)

    if not c_tokens or not r_tokens:
        return {"meteor": 0.0, "f_mean": 0.0, "penalty": 0.0}

    # Unigram alignment
    r_counter = Counter(r_tokens)
    matched_indices = []
    matched_count = 0

    for idx, token in enumerate(c_tokens):
        if r_counter.get(token, 0) > 0:
            matched_count += 1
            r_counter[token] -= 1
            matched_indices.append(idx)

    if matched_count == 0:
        return {"meteor": 0.0, "f_mean": 0.0, "penalty": 0.0}

    precision = float(matched_count) / float(len(c_tokens))
    recall = float(matched_count) / float(len(r_tokens))

    # Harmonic mean weighted towards recall (alpha=0.9)
    f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision) if (recall + 9.0 * precision) > 0 else 0.0

    # Count contiguous chunks of matches
    chunks = 0
    if matched_indices:
        chunks = 1
        for i in range(1, len(matched_indices)):
            if matched_indices[i] != matched_indices[i - 1] + 1:
                chunks += 1

    # Chunk fragmentation penalty: 0.5 * (chunks / matched_count)^3
    penalty = 0.5 * ((float(chunks) / float(matched_count)) ** 3)
    meteor_score = f_mean * (1.0 - penalty)

    return {
        "meteor": round(min(max(meteor_score * 100.0, 0.0), 100.0), 2),
        "f_mean": round(f_mean * 100.0, 2),
        "penalty": round(penalty, 3)
    }


def calculate_semantic_similarity(candidate: str, reference: str) -> Dict[str, Any]:
    """
    Calculate semantic similarity based on token overlap Jaccard index
    and character bigram Dice coefficient.
    """
    c_words = set(tokenize(candidate))
    r_words = set(tokenize(reference))

    if not c_words or not r_words:
        return {"similarity_score": 0.0, "level": "Low"}

    # Word-level Jaccard
    intersection = len(c_words.intersection(r_words))
    union = len(c_words.union(r_words))
    jaccard = float(intersection) / float(union) if union > 0 else 0.0

    # Character bigram similarity for partial word stem alignment
    def get_char_bigrams(text: str):
        cleaned = re.sub(r"\s+", "", text.lower())
        return Counter([cleaned[i:i+2] for i in range(len(cleaned) - 1)])

    c_bg = get_char_bigrams(candidate)
    r_bg = get_char_bigrams(reference)
    bg_overlap = sum(min(cnt, r_bg.get(bg, 0)) for bg, cnt in c_bg.items())
    total_bg = sum(c_bg.values()) + sum(r_bg.values())
    dice = (2.0 * bg_overlap / total_bg) if total_bg > 0 else 0.0

    # Composite similarity score
    composite = (0.6 * jaccard) + (0.4 * dice)
    sim_pct = round(min(max(composite * 100.0, 0.0), 100.0), 2)

    if sim_pct >= 75.0:
        level = "Very High"
    elif sim_pct >= 55.0:
        level = "High"
    elif sim_pct >= 35.0:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "similarity_score": sim_pct,
        "level": level,
        "shared_tokens": list(c_words.intersection(r_words))
    }


def evaluate_caption(
    candidate: str,
    reference: Optional[str] = None,
    image_semantics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive caption quality evaluation pipeline.
    If reference is provided, computes full quantitative benchmarks (BLEU, ROUGE, METEOR, Similarity).
    If reference is not provided, computes linguistic reference-free AI quality score.
    """
    c_tokens = tokenize(candidate)
    word_count = len(c_tokens)

    # Reference-free evaluation metrics
    # 1. Fluency & structure (deduplication, sentence completion)
    unique_words = len(set(c_tokens))
    lexical_diversity = (unique_words / word_count) if word_count > 0 else 0.0
    
    # 2. Semantic alignment with detected scene entities if available
    semantic_coverage = 0.0
    if image_semantics and image_semantics.get("objects"):
        obj_matches = [obj.lower() for obj in image_semantics["objects"] if obj.lower() in candidate.lower()]
        semantic_coverage = (len(obj_matches) / len(image_semantics["objects"])) if image_semantics["objects"] else 0.5

    # 3. Length appropriateness (ideal: 7 to 18 words)
    if 7 <= word_count <= 20:
        length_score = 1.0
    elif word_count < 7:
        length_score = max(0.5, word_count / 7.0)
    else:
        length_score = max(0.7, 1.0 - ((word_count - 20) * 0.03))

    if reference:
        # Full Reference-grounded evaluation
        bleu = calculate_bleu(candidate, reference)
        rouge = calculate_rouge(candidate, reference)
        meteor = calculate_meteor(candidate, reference)
        sim = calculate_semantic_similarity(candidate, reference)

        # Composite quality score weighted across all metric families
        composite_score = (
            0.30 * bleu["bleu_2"] +
            0.35 * rouge["rouge_l"]["f1"] +
            0.25 * meteor["meteor"] +
            0.10 * sim["similarity_score"]
        )
        composite_score = round(min(max(composite_score, 0.0), 100.0), 1)

        if composite_score >= 85.0:
            grade = "Exceptional"
        elif composite_score >= 70.0:
            grade = "High Quality"
        elif composite_score >= 50.0:
            grade = "Good"
        else:
            grade = "Fair"

        return {
            "mode": "reference_grounded",
            "reference_caption": reference,
            "composite_score": composite_score,
            "grade": grade,
            "bleu": bleu,
            "rouge": rouge,
            "meteor": meteor,
            "semantic_similarity": sim,
            "reference_free_stats": {
                "word_count": word_count,
                "lexical_diversity": round(lexical_diversity * 100, 1),
                "length_score": round(length_score * 100, 1)
            }
        }
    else:
        # Reference-free heuristic score (0-100)
        heuristic_score = (
            0.40 * (lexical_diversity * 100.0) +
            0.30 * (length_score * 100.0) +
            0.30 * (min(semantic_coverage * 100.0 + 40.0, 100.0))
        )
        heuristic_score = round(min(max(heuristic_score, 50.0), 98.0), 1)

        if heuristic_score >= 88.0:
            grade = "Exceptional"
        elif heuristic_score >= 76.0:
            grade = "High Quality"
        elif heuristic_score >= 60.0:
            grade = "Good"
        else:
            grade = "Fair"

        return {
            "mode": "reference_free",
            "composite_score": heuristic_score,
            "grade": grade,
            "lexical_diversity": round(lexical_diversity * 100, 1),
            "length_score": round(length_score * 100, 1),
            "semantic_coverage": round(semantic_coverage * 100, 1)
        }
