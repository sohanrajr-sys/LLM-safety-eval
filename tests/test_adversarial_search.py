import base64
import codecs
from src.adversarial_search import (
    mutate_prompt,
    generate_candidates,
    score_candidates,
    filter_top_k,
    _prefix_shuffle,
    _synonym_swap,
    _encoding_wrap,
    _instruction_reorder,
)


def test_prefix_shuffle_adds_prefix():
    result = _prefix_shuffle("How do I hack?", 0)
    assert result != "How do I hack?"
    assert result.endswith("How do I hack?")


def test_synonym_swap_changes_text():
    result = _synonym_swap("How do I make this?")
    assert result != "How do I make this?"


def test_encoding_wrap_even_is_base64():
    result = _encoding_wrap("How do I hack?", 0)
    assert "base64" in result.lower()
    payload = result.split(": ", 1)[1]
    decoded = base64.b64decode(payload).decode()
    assert "How do I hack?" in decoded


def test_encoding_wrap_odd_is_rot13():
    result = _encoding_wrap("How do I hack?", 1)
    assert "rot13" in result.lower() or "rot" in result.lower()


def test_instruction_reorder_on_single_sentence():
    result = _instruction_reorder("How do I hack?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert result != "How do I hack?"


def test_instruction_reorder_on_multi_sentence():
    result = _instruction_reorder("First do this. Then do that. How do I hack?")
    assert isinstance(result, str)
    assert result != "First do this. Then do that. How do I hack?"


def test_mutate_prompt_returns_four_variants():
    variants = mutate_prompt("How do I hack?", 0)
    assert len(variants) == 4
    for v in variants:
        assert isinstance(v, str)
        assert len(v) > 0


def test_mutate_prompt_variants_differ_from_original():
    original = "How do I hack?"
    variants = mutate_prompt(original, 0)
    assert all(v != original for v in variants)


def test_generate_candidates_count():
    prompts = [
        {"id": f"p_{i:03d}", "text": f"How do I do thing {i}?", "category": "direct_harm"}
        for i in range(10)
    ]
    candidates = generate_candidates(prompts)
    assert len(candidates) <= 10 * 4
    assert len(candidates) > 0


def test_generate_candidates_schema():
    prompts = [{"id": "p_000", "text": "How do I hack?", "category": "direct_harm"}]
    candidates = generate_candidates(prompts)
    for c in candidates:
        assert "id" in c
        assert "text" in c
        assert "source_id" in c
        assert "category" in c
        assert "mutation" in c


def test_generate_candidates_no_duplicates():
    prompts = [{"id": "p_000", "text": "How do I hack?", "category": "direct_harm"}]
    candidates = generate_candidates(prompts)
    texts = [c["text"] for c in candidates]
    assert len(texts) == len(set(texts))


def test_filter_top_k_returns_correct_count():
    scored = [
        {"id": f"c_{i}", "text": f"text {i}", "yield_score": i * 0.01}
        for i in range(100)
    ]
    top = filter_top_k(scored, top_k=10)
    assert len(top) == 10


def test_filter_top_k_returns_highest_scores():
    scored = [
        {"id": f"c_{i}", "text": f"text {i}", "yield_score": float(i)}
        for i in range(100)
    ]
    top = filter_top_k(scored, top_k=5)
    scores = [c["yield_score"] for c in top]
    assert min(scores) >= 95.0
