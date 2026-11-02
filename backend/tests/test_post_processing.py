import os
import sys

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.post_processor import (
    remove_special_tokens,
    normalize_spacing_and_punctuation,
    normalize_capitalization,
    ensure_terminal_punctuation,
    clean_caption,
    calculate_confidence,
    get_text_stats
)


def run_post_processing_tests():
    print("==========================================================")
    print("  Milestone 3.7: Caption Post-Processing Test Suite       ")
    print("==========================================================")

    # 1. Special token removal
    print("\n[Test 1] Special token removal")
    raw_tokens = "[CLS] A photo of a cat [SEP] <s> in the garden </s> [PAD]"
    cleaned_tokens = remove_special_tokens(raw_tokens).strip()
    print(f"  Input : '{raw_tokens}'")
    print(f"  Output: '{cleaned_tokens}'")
    assert "[CLS]" not in cleaned_tokens and "[SEP]" not in cleaned_tokens
    assert "<s>" not in cleaned_tokens and "<pad>" not in cleaned_tokens

    # 2. Spacing and punctuation normalization
    print("\n[Test 2] Spacing and punctuation normalization")
    raw_punct = "  a   beautiful   landscape  ,  with mountains  ..  "
    norm_punct = normalize_spacing_and_punctuation(raw_punct)
    print(f"  Input : '{raw_punct}'")
    print(f"  Output: '{norm_punct}'")
    assert "  " not in norm_punct
    assert " ," not in norm_punct
    assert ".." not in norm_punct
    assert norm_punct == "a beautiful landscape, with mountains."

    # 3. Capitalization normalization
    print("\n[Test 3] Capitalization normalization")
    raw_cap = "a young boy playing outside. i think he is happy."
    norm_cap = normalize_capitalization(raw_cap)
    print(f"  Input : '{raw_cap}'")
    print(f"  Output: '{norm_cap}'")
    assert norm_cap.startswith("A")
    assert ". I " in norm_cap

    # 4. Terminal punctuation
    print("\n[Test 4] Terminal punctuation")
    raw_term = "A red car parked on the street"
    norm_term = ensure_terminal_punctuation(raw_term)
    print(f"  Input : '{raw_term}'")
    print(f"  Output: '{norm_term}'")
    assert norm_term.endswith(".")

    # 5. Degraded / Empty input handling
    print("\n[Test 5] Empty / degraded output fallback")
    empty_res = clean_caption("")
    space_res = clean_caption("   ...   ")
    print(f"  Empty input result     : '{empty_res}'")
    print(f"  Whitespace-only result : '{space_res}'")
    assert len(empty_res) > 5
    assert len(space_res) > 5

    # 6. Confidence calculation
    print("\n[Test 6] Confidence scoring")
    conf_high = calculate_confidence(-0.4)
    conf_mod = calculate_confidence(-2.2)
    print(f"  High score (-0.4) -> {conf_high['score']}% ({conf_high['level']})")
    print(f"  Mod score  (-2.2) -> {conf_mod['score']}% ({conf_mod['level']})")
    assert conf_high["score"] >= conf_mod["score"]
    assert conf_high["level"] in ["High", "Good"]

    # 7. Text stats
    print("\n[Test 7] Text stats")
    stats = get_text_stats("A white dog running on the green field.")
    print(f"  Text stats: {stats}")
    assert stats["word_count"] == 8
    assert stats["char_count"] == 39

    print("\n[SUCCESS] All post-processing unit tests passed successfully!")


if __name__ == "__main__":
    run_post_processing_tests()
